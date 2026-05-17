import os
import sys
import torch
import torch.nn.functional as F

# 将项目根目录加入系统路径，确保能顺利导入 src 和 config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.cls_finetune import load_and_modify_cls_model
from src.data.cls_dataloader import get_cls_dataloaders

# 【修改点1】：导入你真实的配置变量 cfg
from config import cfg

# 【修改点2】：将 cfg 对象转换为底层模型改造函数期望的字典格式
# 注意：你的 config 中叫 embedding_dim，底层字典里我们映射为 embed_dim
GPT_CONFIG = {
    'vocab_size': cfg.vocab_size,
    'embed_dim': cfg.embedding_dim,  
    'context_len': cfg.context_len,
    'dropout': cfg.dropout,
    'num_heads': cfg.num_heads,
    'bias': cfg.bias,
    'num_layers': cfg.num_layers
}

def evaluate_accuracy(model, data_loader, device):
    """评估准确率"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)[:, -1, :] # 提取最后一个 Token
            preds = torch.argmax(logits, dim=-1)
            
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / total

def train():
    # 直接使用你 config 中定义的 device
    device = torch.device(cfg.device)
    print(f"🔥 当前运行设备: {device}")
    
    # 1. 准备数据 (请确保这两个 csv 文件真实存在)
    train_csv = "data/cls_raw/train.csv"
    val_csv = "data/cls_raw/validation.csv"
    
    train_loader, val_loader, max_len = get_cls_dataloaders(
        train_path=train_csv,
        val_path=val_csv,
        batch_size=8
    )
    print(f">> 数据集加载完毕。统一序列长度: {max_len}")
    
    # 2. 挂载分类模型
    # 请确认这个路径下真的有你预训练好的权重文件，如果没有，可以先注释掉底层的 load_state_dict 来测通代码
    weights_path = "checkpoints/epoch_3.pth" 
    
    model = load_and_modify_cls_model(weights_path, GPT_CONFIG, num_classes=2)
    model.to(device)
    
    # 3. 设置优化器 (极其重要：仅过滤出被解冻的层)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # 直接使用你 cfg 里定义的学习率
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=0.1)
    
    # 4. 训练循环
    epochs = cfg.epochs
    print("🚀 分类微调正式启动...")
    for epoch in range(epochs):
        model.train()
        
        for step, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            # 取最后一个 Token 获取句子表征
            last_token_logits = logits[:, -1, :] 
            
            loss = F.cross_entropy(last_token_logits, targets)
            loss.backward()
            optimizer.step()
            
            if step % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Step [{step}/{len(train_loader)}] | Loss: {loss.item():.4f}")
        
        # 轮次评估
        train_acc = evaluate_accuracy(model, train_loader, device)
        val_acc = evaluate_accuracy(model, val_loader, device)
        print(f"✨ Epoch {epoch+1} 评估 -> 训练集准确率: {train_acc*100:.2f}% | 验证集准确率: {val_acc*100:.2f}%")
        
    # 5. 保存模型
    # 确保 checkpoints 文件夹存在
    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/classifier_finetuned.pth"
    torch.save(model.state_dict(), save_path)
    print(f"🎉 模型权重已安全保存至: {save_path}")

if __name__ == "__main__":
    train()