"""
清洗获得的数据语料并分词
"""
import tiktoken
import torch
import os
import re

def clean_text(text):
    print("正在进行文本清洗...")
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text

def preprocess_and_save(input_file="./data/raw/mixed_pretrain_test.txt", output_file="./data/processed/pretrain_tokens.pt"):
    print(f"正在读取 {input_file}...")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"找不到文件 {input_file}，请先运行下载脚本。")
        
    with open(input_file, "r", encoding="utf-8") as f:
        txt = f.read()

    original_len = len(txt)
    txt = clean_text(txt)
    cleaned_len = len(txt)
    print(f"清洗完成！清理了 {original_len - cleaned_len:,} 个冗余字符。")

    print("正在进行 BPE 分词 (这可能需要一两分钟)...")
    tokenizer = tiktoken.get_encoding('gpt2')
    token_ids = tokenizer.encode(txt, allowed_special={'<|endoftext|>'})
    print(f"✅ 分词完成！总 Token 数量: {len(token_ids):,}")

    print(f"正在将 Token 转换为张量并固化到 {output_file}...")
    token_tensor = torch.tensor(token_ids, dtype=torch.long)
    torch.save(token_tensor, output_file)
    print("🎉 预处理彻底完成！")

if __name__ == "__main__":
    preprocess_and_save()