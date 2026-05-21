# app.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import tiktoken
from flask import Flask, request, jsonify, render_template

from config import cfg
from src.models.gpt_model import GPTModel
from src.utils.generation import generate_text_simple, generate_temp_topk, text_to_ids, ids_to_text,generate_instruction_response

app = Flask(__name__)

# ── 全局加载模型（只加载一次，不重复加载）──
print("正在加载模型...")
tokenizer = tiktoken.get_encoding("gpt2")

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
    """【新增】：专门用于加载分类微调模型（判别式）"""
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

pretrain_model  = build_model("epoch_3.pth")          # 预训练模型
instruct_model  = build_model("instruct_epoch2.pt")   # 指令微调模型
classify_model  = build_classify_model("classifier_finetuned.pth") # 【新增】：加载你的分类权
print("模型加载完毕，服务启动中...")


# ── 页面 ──
@app.route("/")
def index():
    return render_template("index.html")


# ── 预训练续写接口 ──
@app.route("/api/pretrain", methods=["POST"])
def api_pretrain():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "text 不能为空"}), 400

    idx = text_to_ids(text, tokenizer).to(cfg.device)
    out = generate_temp_topk(
        pretrain_model, idx,
        max_new_tokens=50,
        context_size=cfg.context_len,
        temp=0.8,
        top_k=40,
        eos_id=tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
    )
    result = ids_to_text(out, tokenizer)
    return jsonify({"result": result})


# ── 指令微调接口 ──
@app.route("/api/instruct", methods=["POST"])
def api_instruct():
    data = request.get_json(force=True)
    instruction = data.get("text", "").strip()
    input_text  = data.get("input", "").strip()
    if not instruction:
        return jsonify({"error": "text 不能为空"}), 400

    response = generate_instruction_response(
        instruct_model, tokenizer,
        instruction=instruction,
        input_text=input_text,
        max_new_tokens=50,
        context_size=cfg.context_len,
        temperature=0.6,
        top_k=20,
        repetition_penalty=1.3,
        device=cfg.device
    )
    return jsonify({"result": response})

# ── 分类接口（预留，由队友对接）──
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