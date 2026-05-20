# app.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
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

pretrain_model  = build_model("epoch_3.pth")          # 预训练模型
instruct_model  = build_model("instruct_epoch1.pt")   # 指令微调模型
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
    return jsonify({"result": "分类接口待接入"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)