"""
配置文件
"""
import torch

class Config:
    vocab_size=50257  # 词汇表大小
    context_len=1024  # 上下文长度

    embedding_dim=768  # 嵌入维度
    num_heads=12  # 多头注意力头数
    num_layers=12  # 层数量
    lr=0.0004  # 学习率
    epochs=10  # 训练轮数

    dropout=0.1  # dropout概率
    bias=False  # 是否使用偏置项

    device='cuda' if torch.cuda.is_available() else 'cpu'  # 设备类型



cfg=Config()
