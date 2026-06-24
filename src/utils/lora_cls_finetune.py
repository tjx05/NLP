import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from src.models.gpt_model import GPTModel

def load_and_modify_lora_cls_model(weights_path, config, lora_r=8, lora_alpha=32, lora_dropout=0.05, num_classes=2):
    """
    加载自定义GPT模型，并使用 PEFT(LoRA) 将其改造成参数高效的分类器。
    
    参数:
        weights_path: 预训练基座模型权重的路径
        config: GPT 模型的基础超参数字典
        lora_r: LoRA 的秩 (Rank)，默认 8
        lora_alpha: LoRA 的缩放系数，默认 32
        lora_dropout: LoRA 层的 dropout，默认 0.05
        num_classes: 分类任务的类别数，默认为 2
    """
    print("1. 正在初始化 GPT 基座模型骨架...")
    model = GPTModel(
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        context_len=config['context_len'],
        dropout=config['dropout'],
        num_heads=config['num_heads'],
        bias=config['bias'],
        num_layers=config['num_layers']
    )
    
    print(f"2. 正在加载预训练权重: {weights_path}")
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    print(">> 预训练基座权重加载成功！")

    print("3. 开始执行 LoRA 架构重构...")
    # A. 替换原本的生成式输出层为二分类头
    # 【注意】这一步必须在 get_peft_model 之前完成
    model.output = nn.Linear(config['embed_dim'], num_classes, bias=False) 

    # B. 配置 LoRA 挂载规则
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        # 精准定位到 attention.py 中定义的独立线性层
        target_modules=["w_q", "w_k", "w_v"], 
        
        # modules_to_save 强制要求 PEFT 保留并训练分类头
        modules_to_save=["output"], 
        lora_dropout=lora_dropout,
        bias="none",
    )
    
    # C. 使用 PEFT 包装模型，自动冻结原参数并注入旁路矩阵
    model = get_peft_model(model, lora_config)
    
    # 打印参数对比日志，直观展示参数压缩效果
    print("\n" + "="*50)
    model.print_trainable_parameters()
    print("="*50 + "\n")
    
    return model