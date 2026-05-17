import torch
import torch.nn as nn
from src.models.gpt_model import GPTModel

def load_and_modify_cls_model(weights_path, config, num_classes=2):
    """
    加载自定义预训练GPT模型，并将其改造成专属分类器
    """
    print("1. 正在初始化 GPT 模型骨架...")
    # 这里直接使用传入的 config 字典进行实例化
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
    print(">> 预训练权重加载成功！")

    print("3. 开始执行分类器架构改造与参数冻结...")
    # A. 冻结整个骨干网络
    for param in model.parameters():
        param.requires_grad = False
        
    # B. 替换原有的词表输出层 (你源码中定义为 self.output)
    # 新的线性层默认 requires_grad=True
    model.output = nn.Linear(config['embed_dim'], num_classes, bias=False) 
    
    # C. 解冻最后一层 Transformer 块和最后的 LayerNorm
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True
        
    print(">> 改造完成！当前仅最后一层 Transformer Block、Final_Norm 以及新分类头参与梯度更新。")
    return model