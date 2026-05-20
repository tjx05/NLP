# scripts/run_instruction_test.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import tiktoken

from config import cfg
from src.models.gpt_model import GPTModel
from src.utils.generation import generate_instruction_response

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ========== 测试用例 ==========
TEST_CASES = [
    {"instruction": "Expand the following sentence into a paragraph",     "input": "The scientist made an important discovery."},
    {"instruction": "Expand the following sentence into a paragraph",     "input": "She walked into the room and everything changed."},
    {"instruction": "What is machine learning?",                          "input": ""},
    {"instruction": "What is the difference between AI and human intelligence?", "input": ""},
    {"instruction": "Write a short sentence about the mountains",         "input": ""},
    {"instruction": "Write a short sentence about friendship",            "input": ""},
    {"instruction": "Explain what photosynthesis is",                     "input": ""},
    {"instruction": "Explain why the sky is blue",                        "input": ""},
]


def load_model(checkpoint_name):
    model = GPTModel(
        vocab_size=cfg.vocab_size,
        embed_dim=cfg.embedding_dim,
        context_len=cfg.context_len,
        dropout=cfg.dropout,
        num_heads=cfg.num_heads,
        bias=cfg.bias,
        num_layers=cfg.num_layers
    ).to(cfg.device)

    checkpoint_path = os.path.join(BASE_DIR, "checkpoints", checkpoint_name)
    checkpoint = torch.load(checkpoint_path, map_location=cfg.device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"Loaded: {checkpoint_path}\n")
    return model


def run_test(model, tokenizer, label):
    print("=" * 60)
    print(f"  Checkpoint: {label}")
    print("=" * 60)
    for item in TEST_CASES:
        response = generate_instruction_response(
            model, tokenizer,
            instruction=item["instruction"],
            input_text=item["input"],
            max_new_tokens=60,
            context_size=cfg.context_len,
            temperature=0.6,  # 从0.3调高
            top_k=50,  # 从20调大
            repetition_penalty=1.3,  # 新增
            device=cfg.device
        )
        print(f"[Instruction] {item['instruction']}")
        if item["input"]:
            print(f"[Input]       {item['input']}")
        print(f"[Response]    {response}")
        print("-" * 60)
    print()



def load_alpaca_cleaned_data(path="data/raw/alpaca_cleaned.json"):
    """
    Alpaca-Cleaned: 社区清洗版，约52000条，去除了噪声和重复样本
    比原版Alpaca质量明显更高
    """
    import urllib.request, os, json
    url = "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/main/alpaca_data_cleaned.json"
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("Downloading Alpaca-Cleaned...")
        urllib.request.urlretrieve(url, path)
        print(f"Downloaded to {path}")
    with open(path, "r") as f:
        return json.load(f)


def load_dolly_data(path="data/raw/dolly_data.json"):
    """
    Databricks Dolly-15k: 15000条人工标注，指令类型多样
    包含问答、摘要、创意写作、信息提取等8种任务类型
    格式需转换成统一的 instruction/input/output 格式
    """
    import os, json
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("Downloading Dolly-15k...")
        from datasets import load_dataset
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
        converted = []
        for item in ds:
            converted.append({
                "instruction": item["instruction"],
                "input":       item["context"],   # dolly叫context
                "output":      item["response"]
            })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(converted, f, ensure_ascii=False, indent=2)
        print(f"Saved to {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_combined_data(alpaca_limit=5000, dolly_limit=5000):
    """
    混合数据集：Alpaca-Cleaned + Dolly-15k
    两个数据集互补，覆盖更多指令类型
    """
    import random
    alpaca = load_alpaca_cleaned_data()[:alpaca_limit]
    dolly  = load_dolly_data()[:dolly_limit]
    combined = alpaca + dolly
    random.shuffle(combined)
    print(f"Combined dataset: {len(alpaca)} alpaca + {len(dolly)} dolly = {len(combined)} total")
    return combined


def main():
    tokenizer = tiktoken.get_encoding("gpt2")

    # 对比两个 checkpoint

    for ckpt_name in ["instruct_epoch1.pt", "instruct_epoch2.pt","instruct_epoch3.pt"]:
        model = load_model(ckpt_name)
        run_test(model, tokenizer, label=ckpt_name)

    """
    # 最小数据集的第二轮c
    model = load_model("instruct_epoch2_simple.pt")
    run_test(model, tokenizer, label="instruct_epoch2_simple.pt")
    """


if __name__ == "__main__":
    main()