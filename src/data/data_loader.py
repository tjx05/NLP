"""
清洗获得的数据语料并分词-5B工业化改进版
"""
import tiktoken
import os
import re
from tqdm import tqdm
import numpy as np

def clean_text(text):
    # 稍微减少打印次数，否则 1 亿行数据会刷屏
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text

def preprocess_and_save():
    input_dir = "./data/raw"
    output_dir = "./data/processed"
    output_bin = os.path.join(output_dir, "train.bin")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 自动获取所有 txt 文件并按序号排序
    files = sorted([f for f in os.listdir(input_dir) if f.endswith('.txt')],
                   key=lambda x: int(re.findall(r'\d+', x)[0]))
    
    tokenizer = tiktoken.get_encoding('gpt2')
    total_tokens = 0

    print(f"检测到 {len(files)} 个分块文件，开始追加预处理...")

    # 使用 'wb' (二进制追加) 模式打开文件
    with open(output_bin, "wb") as bin_file:
        for file_name in tqdm(files, desc="分词进度"):
            file_path = os.path.join(input_dir, file_name)
            
            with open(file_path, "r", encoding="utf-8") as f:
                # 每次只读取一个 1GB 的文件，内存非常安全
                txt = f.read()

            # 执行清洗
            txt = clean_text(txt)

            # 执行分词
            token_ids = tokenizer.encode(txt, allowed_special={'<|endoftext|>'})
            total_tokens += len(token_ids)

            # 转换为 uint16 并直接写入二进制流
            # uint16 占用 2 字节，比 torch.long (8 字节) 节省 75% 的磁盘空间
            token_np = np.array(token_ids, dtype=np.uint16)
            bin_file.write(token_np.tobytes())

    print("\n" + "="*30)
    print(f"预处理彻底完成！")
    print(f"总 Token 数量: {total_tokens:,}")
    print(f"最终产物: {output_bin}")
    print(f"文件大小: {os.path.getsize(output_bin) / (1024**3):.2f} GB")
    print("="*30)

if __name__ == "__main__":
    preprocess_and_save()