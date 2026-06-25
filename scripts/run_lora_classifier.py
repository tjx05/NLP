import os
import sys
import time # 用于记录每轮训练耗时
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix # 用于深度评估

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.lora_cls_finetune import load_and_modify_lora_cls_model
from src.data.cls_dataloader import get_cls_dataloaders
from config import cfg

# ================= 基础基座配置 =================
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
    """评估模型在给定数据集上的准确率"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)[:, -1, :] 
            preds = torch.argmax(logits, dim=-1)
            
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / total

# 深度评估函数：输出 F1, Precision, Recall 和 混淆矩阵
def evaluate_comprehensive(model, data_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)[:, -1, :] 
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    # 假设 0: Normal(正常短信), 1: Spam(垃圾短信)
    report = classification_report(all_targets, all_preds, target_names=['Normal', 'Spam'], digits=4)
    cm = confusion_matrix(all_targets, all_preds)
    
    return report, cm

def train():
    device = torch.device(cfg.device)
    print(f"当前运行设备: {device}")
    
    # 1. 准备数据 
    train_csv = "data/cls_raw/train.csv"
    val_csv = "data/cls_raw/validation.csv"
    
    train_loader, val_loader, max_len = get_cls_dataloaders(
        train_path=train_csv,
        val_path=val_csv,
        batch_size=8
    )
    print(f">> 数据集加载完毕。统一序列长度: {max_len}")
    
    # 2. 挂载带 LoRA 的分类模型
    weights_path = "checkpoints/epoch_3.pth" 
    
    model = load_and_modify_lora_cls_model(
        weights_path=weights_path, 
        config=GPT_CONFIG, 
        lora_r=8, 
        lora_alpha=32, 
        lora_dropout=0.05, 
        num_classes=2
    )
    model.to(device)
    
    # 3. 设置优化器
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=0.1)
    
    # 指标记录列表
    step_losses = []
    global_steps = []
    train_accuracies = []
    val_accuracies = []
    global_step_counter = 0

    # 4. 训练核心循环
    epochs = cfg.epochs
    print(" LoRA 垃圾短信分类微调正式启动...")
    for epoch in range(epochs):
        epoch_start_time = time.time() # 记录本轮开始时间
        model.train()
        
        for step, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            last_token_logits = logits[:, -1, :] 
            
            loss = F.cross_entropy(last_token_logits, targets)
            loss.backward()
            optimizer.step()
            
            step_losses.append(loss.item())
            global_steps.append(global_step_counter)
            global_step_counter += 1
            
            if step % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Step [{step}/{len(train_loader)}] | Loss: {loss.item():.4f}")
        
        # 轮次评估
        train_acc = evaluate_accuracy(model, train_loader, device)
        val_acc = evaluate_accuracy(model, val_loader, device)
        
        train_accuracies.append(train_acc * 100)
        val_accuracies.append(val_acc * 100)
        
        epoch_end_time = time.time() # 记录本轮结束时间
        # 打印信息加入了单轮耗时
        print(f" Epoch {epoch+1} 评估 -> 训练集准确率: {train_acc*100:.2f}% | 验证集准确率: {val_acc*100:.2f}% | 耗时: {epoch_end_time - epoch_start_time:.2f} 秒")
        
    # 5. 保存权重
    save_dir = "checkpoints/lora_finetuned_spam"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir) 
    print(f" LoRA 轻量级权重已安全保存至: {save_dir}")

    # ================= 深度评估报告输出 =================
    print("\n" + "="*50)
    print(" LoRA 微调：最终验证集深度评估报告 (Validation Report)")
    print("="*50)
    final_report, final_cm = evaluate_comprehensive(model, val_loader, device)
    print(final_report)
    print("混淆矩阵 (Confusion Matrix):")
    print("[[True Normal, False Spam]")
    print(" [False Normal, True Spam]]")
    print(final_cm)
    print("="*50 + "\n")

    # ================= 图表可视化部分 =================
    print(" 正在生成 LoRA 训练指标可视化图表...")
    os.makedirs("output", exist_ok=True)
    
    plt.figure(figsize=(12, 5))

    # 子图 1: Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(global_steps, step_losses, label='LoRA Raw Loss', color='#1f77b4', alpha=0.3)
    
    window_size = 10
    if len(step_losses) >= window_size:
        smoothed_losses = np.convolve(step_losses, np.ones(window_size)/window_size, mode='valid')
        smoothed_steps = global_steps[window_size-1:]
        plt.plot(smoothed_steps, smoothed_losses, label='Smoothed Loss', color='#d62728', linewidth=2)

    plt.xlabel('Global Steps')
    plt.ylabel('Cross Entropy Loss')
    plt.title('LoRA Training Loss (Spam Detection)')
    plt.ylim(0, 2.5) 
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    # 子图 2: Accuracy 曲线
    plt.subplot(1, 2, 2)
    epochs_range = range(1, epochs + 1)
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy', marker='o', color='#ff7f0e', linewidth=2)
    plt.plot(epochs_range, val_accuracies, label='Val Accuracy', marker='s', color='#2ca02c', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('LoRA Train & Val Accuracy')
    plt.xticks(epochs_range)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plot_save_path = "output/lora_cls_training.png"
    plt.savefig(plot_save_path, dpi=300)
    plt.close()
    print(f"LoRA 可视化图表已成功保存至: {plot_save_path}")

if __name__ == "__main__":
    train()