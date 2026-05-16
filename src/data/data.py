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
    print("✅ 5B语料全量拉取完成！")

if __name__ == "__main__":
    build_5b_dataset()