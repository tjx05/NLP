import os
import sys
import torch
import torch.nn as nn
import tiktoken

# 将项目根目录加入系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.gpt_model import GPTModel
from config import cfg

def load_finetuned_model(weights_path, device):
    """
    加载微调后的分类器模型
    """
    print("1. 正在初始化模型骨架...")
    model = GPTModel(
        vocab_size=cfg.vocab_size,
        embed_dim=cfg.embedding_dim,  
        context_len=cfg.context_len,
        dropout=cfg.dropout,
        num_heads=cfg.num_heads,
        bias=cfg.bias,
        num_layers=cfg.num_layers
    )
    
    # 先将模型改造成二分类结构，再加载微调后的权重
    model.output = nn.Linear(cfg.embedding_dim, 2, bias=False)
    
    print(f"2. 正在加载微调后的权重: {weights_path} ...")
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval() # 切换到评估模式，关闭 Dropout
    print(">> 模型加载就绪！\n")
    return model

def classify_text(text, model, tokenizer, device, max_length=120, pad_token_id=50256):
    """
    对单条文本进行分类预测
    """
    # 文本转 Token ID
    input_ids = tokenizer.encode(text)
    
    #截断或填充至与训练时一致的长度 (120)
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
    else:
        input_ids += [pad_token_id] * (max_length - len(input_ids))
        
    # 转为张量并增加 Batch 维度: [1, seq_len]
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    # 前向传播
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :] # 取出最后一个 Token 的输出
        predicted_label = torch.argmax(logits, dim=-1).item()
        
    return "垃圾信息 (Spam) " if predicted_label == 1 else "正常信息 (Ham) "

def main():
    device = torch.device(cfg.device)
    weights_path = "checkpoints/classifier_finetuned.pth"
    
    # 检查权重是否存在
    if not os.path.exists(weights_path):
        print(f"找不到权重文件 {weights_path}，请确认是否已成功运行 run_classifier.py")
        return

    # 加载模型与分词器
    model = load_finetuned_model(weights_path, device)
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # ========== 开始测试 ==========
    print("="*40)
    print("欢迎使用垃圾文本分类测试 (输入 'quit' 退出)")
    print("="*40)
    
    # 预设几个典型的测试用例
    test_cases = [
        "Hey, just wanted to check if we're still on for dinner tonight? Let me know!", 
        "CONGRATULATIONS! You are a winner! You have been specially selected to receive $1000 cash. Reply WIN to claim.",
        "Can you send me the report by 5 PM? Thanks."
    ]
    
    for text in test_cases:
        result = classify_text(text, model, tokenizer, device)
        print(f"文本: {text}")
        print(f"预测: {result}\n")
        
    # 允许手动输入测试
    while True:
        user_input = input("请输入你想测试的英文句子 (或输入 'quit' 退出): \n> ")
        if user_input.lower() in ['quit', 'q', 'exit']:
            break
        if not user_input.strip():
            continue
            
        result = classify_text(user_input, model, tokenizer, device)
        print(f"--> 预测结果: {result}\n")

if __name__ == "__main__":
    main()