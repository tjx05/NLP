import os
# 设置镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset
from tqdm import tqdm  # 导入进度条库

def build_5b_dataset(target_tokens=5_000_000_000):
    output_dir = "./data/raw"
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # 1个Token约3.5字节
    target_total_bytes = target_tokens * 3.5
    
    configs = [
        {"path": "HuggingFaceFW/fineweb-edu", "name": "sample-10BT", "split": "train", "ratio": 0.50},
        {"path": "roneneldan/TinyStories", "split": "train", "ratio": 0.30}, 
        {"path": "wikimedia/wikipedia", "name": "20231101.en", "split": "train", "ratio": 0.20}
    ]

    file_idx = 1
    current_file_size = 0
    MAX_FILE_SIZE = 1024 * 1024 * 1024  # 1GB 一个分块
    
    # 开启总进度条 (以 GB 为单位)
    total_pbar = tqdm(total=target_total_bytes, unit='B', unit_scale=True, desc="总进度")

    f = open(os.path.join(output_dir, f"train_part_{file_idx}.txt"), "w", encoding="utf-8")

    for config in configs:
        portion_bytes = target_total_bytes * config['ratio']
        downloaded_bytes = 0
        
        # 加载数据集
        ds = load_dataset(config['path'], name=config.get('name'), split=config['split'], streaming=True)
        
        for entry in ds:
            text = entry.get('text') or entry.get('story') or ""
            if not text.strip(): continue
            
            content = text + "\n<|endoftext|>\n"
            line_bytes = len(content.encode('utf-8'))
            
            # 自动分块逻辑
            if current_file_size + line_bytes > MAX_FILE_SIZE:
                f.close()
                file_idx += 1
                f = open(os.path.join(output_dir, f"train_part_{file_idx}.txt"), "w", encoding="utf-8")
                current_file_size = 0

            f.write(content)
            current_file_size += line_bytes
            downloaded_bytes += line_bytes
            
            # 更新进度条
            total_pbar.update(line_bytes)
            
            if downloaded_bytes >= portion_bytes:
                break
                
    f.close()
    total_pbar.close()
    print("5B语料全量拉取完成！")

def load_instruction_data(path="D:/NLP/NLP/data/raw/instruction_data.json"):
    #下载并加载指令数据集
    import urllib.request, os, json
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path)
        print(f"Downloaded to {path}")
    with open(path, "r") as f:
        return json.load(f)

def load_alpaca_data(path="data/raw/alpaca_data.json"):
    """斯坦福 Alpaca 完整数据集，52000条"""
    import urllib.request, os, json
    url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path)
        print(f"Downloaded to {path}")
    with open(path, "r") as f:
        return json.load(f)



def load_alpaca_cleaned_data(path="data/raw/alpaca_cleaned.json"):
    """
    Alpaca-Cleaned: 社区清洗版，约52000条，去除了噪声和重复样本
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

if __name__ == "__main__":
    build_5b_dataset()

