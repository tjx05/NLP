# app.py
import sys
import os

import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import json
import torch
import tiktoken
from datetime import datetime
from flask import Flask, request, jsonify, render_template

from config import cfg
from src.models.gpt_model import GPTModel
from src.utils.generation import generate_temp_topk, text_to_ids, ids_to_text, generate_instruction_response
from src.utils.generation import generate_instruction_response, generate_json_response

app = Flask(__name__)

# ── 历史记录文件路径 ──
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "output", "chat_history.json")

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"pretrain": [], "classify": [], "instruct": []}

def save_history(history):
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# ── 加载模型 ──
print("正在加载模型...")
tokenizer = tiktoken.get_encoding("gpt2")
EOS_ID = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

def build_model(checkpoint_name):
    model = GPTModel(
        vocab_size=cfg.vocab_size,
        embed_dim=cfg.embedding_dim,
        context_len=cfg.context_len,
        dropout=cfg.dropout,
        num_heads=cfg.num_heads,
        bias=cfg.bias,
        num_layers=cfg.num_layers
    ).to(cfg.device)
    path = os.path.join(os.path.dirname(__file__), "checkpoints", checkpoint_name)
    ckpt = torch.load(path, map_location=cfg.device, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()
    print(f"  已加载: {path}")
    return model


def build_classify_model(checkpoint_name):
    """专门用于加载分类微调模型（判别式）"""
    model = GPTModel(
        vocab_size=cfg.vocab_size,
        embed_dim=cfg.embedding_dim,
        context_len=cfg.context_len,
        dropout=cfg.dropout,
        num_heads=cfg.num_heads,
        bias=cfg.bias,
        num_layers=cfg.num_layers
    ).to(cfg.device)

    # 核心改造：将生成头切断，换成二分类头
    model.output = nn.Linear(cfg.embedding_dim, 2, bias=False).to(cfg.device)

    path = os.path.join(os.path.dirname(__file__), "checkpoints", checkpoint_name)
    ckpt = torch.load(path, map_location=cfg.device, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()
    print(f"  已加载分类模型: {path}")
    return model

pretrain_model = build_model("epoch_3.pth")
instruct_model = build_model("begin_json_epoch5.pt")
classify_model  = build_classify_model("classifier_finetuned.pth") # 【新增】：加载你的分类权
print("模型加载完毕，服务启动中...")


def parse_params(data):
    """从请求里读取前端传来的参数，不传则用默认值"""
    return {
        "max_new_tokens":     int(data.get("max_new_tokens",     50)),
        "temperature":        float(data.get("temperature",      0.6)),
        "top_k":              int(data.get("top_k",              40)),
        "repetition_penalty": float(data.get("repetition_penalty", 1.3)),
    }


# ── 页面 ──
@app.route("/")
def index():
    return render_template("index.html")


# ── 历史记录接口 ──
@app.route("/api/history", methods=["GET"])
def api_history():
    return jsonify(load_history())

@app.route("/api/history/save", methods=["POST"])
def api_save_history():
    data  = request.get_json(force=True)
    panel = data.get("panel")
    role  = data.get("role")
    text  = data.get("text")
    if not panel or not role or text is None:
        return jsonify({"ok": False}), 400
    history = load_history()
    if panel not in history:
        history[panel] = []
    history[panel].append({
        "role": role,
        "text": text,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    save_history(history)
    return jsonify({"ok": True})

@app.route("/api/history/clear", methods=["POST"])
def api_clear_history():
    data  = request.get_json(force=True)
    panel = data.get("panel", None)
    history = load_history()
    if panel:
        history[panel] = []
    else:
        history = {"pretrain": [], "classify": [], "instruct": []}
    save_history(history)
    return jsonify({"ok": True})


# ── 预训练续写接口 ──
@app.route("/api/pretrain", methods=["POST"])
def api_pretrain():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "text 不能为空"}), 400

    p = parse_params(data)
    idx = text_to_ids(text, tokenizer).to(cfg.device)
    out = generate_temp_topk(
        pretrain_model, idx,
        max_new_tokens=p["max_new_tokens"],
        context_size=cfg.context_len,
        temp=p["temperature"],
        top_k=p["top_k"],
        eos_id=EOS_ID,
        repetition_penalty=p["repetition_penalty"],
    )
    result = ids_to_text(out, tokenizer)
    return jsonify({"result": result})


# ── 指令微调接口 ──

@app.route("/api/instruct", methods=["POST"])
def api_instruct():
    data        = request.get_json(force=True)
    instruction = data.get("text", "").strip()
    input_text  = data.get("input", "").strip()
    if not instruction:
        return jsonify({"error": "text 不能为空"}), 400

    p = parse_params(data)
    response = generate_json_response(
        instruct_model, tokenizer,
        instruction=instruction,
        input_text=input_text,
        max_new_tokens=p["max_new_tokens"],
        context_size=cfg.context_len,
        temperature=p["temperature"],
        top_k=p["top_k"],
        repetition_penalty=p["repetition_penalty"],
        device=cfg.device,
    )
    return jsonify({"result": response})


# ── 分类接口──
@app.route("/api/classify", methods=["POST"])
def api_classify():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "text 不能为空"}), 400

    # 1. 文本转 Token ID
    input_ids = tokenizer.encode(text)

    # 2. 截断或向右填充至 120
    max_length = 120
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
    else:
        input_ids += [50256] * (max_length - len(input_ids))

    # 3. 转张量并送入设备
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=cfg.device).unsqueeze(0)

    # 4. 推理取最后一个 Token 的输出
    with torch.no_grad():
        logits = classify_model(input_tensor)[:, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        pred_label = torch.argmax(logits, dim=-1).item()

    # 5. 格式化输出字符串给前端
    ham_prob = probs[0].item() * 100
    spam_prob = probs[1].item() * 100

    if pred_label == 1:
        res_str = f"🚫 垃圾/钓鱼信息 (Spam)\n\n【底层置信度】\n正常: {ham_prob:.2f}%\n垃圾: {spam_prob:.2f}%"
    else:
        res_str = f"✅ 正常安全信息 (Ham)\n\n【底层置信度】\n正常: {ham_prob:.2f}%\n垃圾: {spam_prob:.2f}%"

    return jsonify({"result": res_str})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)