import os
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt # 【新增】导入绘图库

# 将项目根目录加入系统路径，确保能顺利导入 src 和 config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.cls_finetune import load_and_modify_cls_model
from src.data.cls_dataloader import get_cls_dataloaders
from config import cfg

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
    device = torch.device(cfg.device)
    print(f"🔥 当前运行设备: {device}")
    
    # 1. 准备数据
    train_csv = "data/cls_raw/train.csv"
    val_csv = "data/cls_raw/validation.csv"
    
    train_loader, val_loader, max_len = get_cls_dataloaders(
        train_path=train_csv,
        val_path=val_csv,
        batch_size=8
    )
    print(f">> 数据集加载完毕。统一序列长度: {max_len}")
    
    # 2. 挂载分类模型
    weights_path = "checkpoints/epoch_3.pth" 
    model = load_and_modify_cls_model(weights_path, GPT_CONFIG, num_classes=2)
    model.to(device)
    
    # 3. 设置优化器
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=0.1)
    
    # 定义用于记录训练指标的列表
    step_losses = []        # 记录每一步的 Loss
    global_steps = []       # 记录全局步数 (作为 Loss 图的 X 轴)
    train_accuracies = []   # 记录每轮的训练集准确率
    val_accuracies = []     # 记录每轮的验证集准确率
    global_step_counter = 0

    # 4. 训练循环
    epochs = cfg.epochs
    print("🚀 分类微调正式启动...")
    for epoch in range(epochs):
        model.train()
        
        for step, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            last_token_logits = logits[:, -1, :] 
            
            loss = F.cross_entropy(last_token_logits, targets)
            loss.backward()
            optimizer.step()
            
            # 记录每一个 step 的 Loss，方便画出平滑曲线
            step_losses.append(loss.item())
            global_steps.append(global_step_counter)
            global_step_counter += 1
            
            if step % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Step [{step}/{len(train_loader)}] | Loss: {loss.item():.4f}")
        
        # 轮次评估
        train_acc = evaluate_accuracy(model, train_loader, device)
        val_acc = evaluate_accuracy(model, val_loader, device)
        
        # 记录每轮的准确率 (转换为百分比)
        train_accuracies.append(train_acc * 100)
        val_accuracies.append(val_acc * 100)
        
        print(f"✨ Epoch {epoch+1} 评估 -> 训练集准确率: {train_acc*100:.2f}% | 验证集准确率: {val_acc*100:.2f}%")
        
    # 5. 保存模型权重
    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/classifier_finetuned1.pth"
    torch.save(model.state_dict(), save_path)
    print(f"🎉 模型权重已安全保存至: {save_path}")

    # ================= 绘图与保存逻辑 =================
    print("📊 正在生成训练指标可视化图表...")
    os.makedirs("output", exist_ok=True) # 确保 output 文件夹存在
    
    # 创建一个 1行2列 的画板，尺寸为 12x5 英寸
    plt.figure(figsize=(12, 5))

    # ================= 子图 1 (Loss 曲线) =================
    plt.subplot(1, 2, 1)
    
    # 1. 绘制原始震荡曲线，但把透明度调低作为背景 (alpha=0.3)
    plt.plot(global_steps, step_losses, label='Raw Loss', color='#1f77b4', alpha=0.3)
    
    # 2. 计算并绘制滑动平均平滑曲线 (业界标配)
    import numpy as np
    window_size = 10 # 平滑窗口大小，数值越大曲线越平滑
    if len(step_losses) >= window_size:
        # 使用卷积计算滑动平均
        smoothed_losses = np.convolve(step_losses, np.ones(window_size)/window_size, mode='valid')
        smoothed_steps = global_steps[window_size-1:]
        # 用醒目的红色粗线画出平滑后的主趋势
        plt.plot(smoothed_steps, smoothed_losses, label='Smoothed Loss', color='#d62728', linewidth=2)

    plt.xlabel('Global Steps')
    plt.ylabel('Cross Entropy Loss')
    plt.title('Training Loss over Steps')
    
    # 3. 强制拉大纵坐标的显示范围 (压缩视觉震荡感)
    # 你可以根据实际跑出的最大 Loss 灵活调整这里的上限，比如 2.0 或 2.5
    plt.ylim(0, 2.5) 
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    # ================= 子图 1 (Loss 曲线) =================

    # 子图 2: 绘制 Accuracy 曲线 (横坐标是 Epoch) 保持不变
    plt.subplot(1, 2, 2)
    epochs_range = range(1, epochs + 1)
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy', marker='o', color='#ff7f0e', linewidth=2)
    plt.plot(epochs_range, val_accuracies, label='Val Accuracy', marker='s', color='#2ca02c', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.xticks(epochs_range) # 强制 x 轴显示整数轮次
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    # 自动调整布局防遮挡，并以 300 DPI 高清保存
    plt.tight_layout()
    plot_save_path = "output/cls_training.png"
    plt.savefig(plot_save_path, dpi=300)
    plt.close() # 关闭画板，释放内存
    print(f"🖼️ 可视化图表已成功保存至: {plot_save_path}")

if __name__ == "__main__":
    train()