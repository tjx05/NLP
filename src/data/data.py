"""
获得数据语料
"""
from datasets import load_dataset
import os

configs = [
    {"path": "HuggingFaceFW/fineweb-edu", "name": "sample-10BT", "split": "train", "ratio": 0.50},
    {"path": "roneneldan/TinyStories", "split": "train", "ratio": 0.30}, 
    {"path": "wikimedia/wikipedia", "name": "20231101.en", "split": "train", "ratio": 0.20}
]

def build_mixed_test_set(target_size_mb=500):
    output_file = "./data/raw/mixed_pretrain_test.txt"
    target_bytes = target_size_mb * 1024 * 1024
    
    print(f"开始构建混合测试集，总目标大小：{target_size_mb}MB")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for config in configs:
            portion_bytes = target_bytes * config['ratio']
            current_bytes = 0
            
            print(f"正在流式下载 {config['path']}...")
            
            ds = load_dataset(
                config['path'], 
                name=config.get('name'), 
                split=config['split'], 
                streaming=True
            )
            
            for entry in ds:
                text = entry.get('text') or entry.get('story') or ""
                if not text.strip(): continue
                
                f.write(text + "\n<|endoftext|>\n")
                current_bytes += len(text.encode('utf-8'))
                
                if current_bytes >= portion_bytes:
                    break
                    
    print(f"✅ 构建完成！混合语料已存至: {output_file}")
    return output_file

if __name__ == "__main__":
    build_mixed_test_set(500)