# scripts/run_instruction_fintune.py
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from functools import partial

import tiktoken
import matplotlib.pyplot as plt
import matplotlib
import json  # 引入 json 模块

matplotlib.use('Agg')  # 非交互模式，保存图片不弹窗

from config import cfg
from src.models.gpt_model import GPTModel
from src.data.instruction_dataset import InstructionDataset, collate_fn

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 🎯 1. 权重加载路径：指向你上一次微调出来的模型权重文件
checkpoint_path = os.path.join(BASE_DIR, "checkpoints", "instruct_epoch1.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# 🎯 2. 数据集路径定义
PATH_NEW_JSON = r"D:\NLP\NLP\data\instrc_raw\instrc_rawfinal_sft_train_dataset.json"  # 新洗好的JSON提取数据
PATH_OLD_SFT = r"D:\NLP\NLP\data\raw\old_instruct.json"  # 刚刚裁剪出的几百条老数据


def warmup_scheduler(optimizer, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


def plot_losses(train_losses, val_losses, step_losses, save_dir):
    """绘制训练过程图：左图step级loss曲线，右图epoch级train/val对比"""
    os.makedirs(save_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：每200步的loss变化
    ax1.plot(step_losses, color='steelblue', linewidth=1.2, alpha=0.8)
    ax1.set_title("Training Loss (per 200 steps)", fontsize=13)
    ax1.set_xlabel("Record Index (every 200 steps)")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 右图：每个epoch的train vs val loss
    epochs = range(1, len(train_losses) + 1)
    ax2.plot(epochs, train_losses, marker='o', label='Train Loss', color='steelblue')
    ax2.plot(epochs, val_losses, marker='s', label='Val Loss', color='tomato')
    ax2.set_title("Train vs Val Loss (per epoch)", fontsize=13)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "instruct_loss_curve.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Loss curve saved to: {save_path}")


def train():
    device = cfg.device
    print(f"Using device: {device}")

    # 1. 加载 tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # 2. 🎯 改造数据加载逻辑：加载新数据 + 缝合老数据
    print("正在加载新洗好的 JSON 提取数据...")
    with open(PATH_NEW_JSON, "r", encoding="utf-8") as f:
        new_json_data = json.load(f)

    print("正在加载裁剪后的老微调数据...")
    with open(PATH_OLD_SFT, "r", encoding="utf-8") as f:
        old_sft_data = json.load(f)

    # 完美缝合
    data = new_json_data + old_sft_data
    print(f"📊 数据缝合完毕！新数据: {len(new_json_data)} 条 | 老数据: {len(old_sft_data)} 条 | 总计: {len(data)} 条")

    # 依然保留你们原代码里的过滤机制（可以防止空数据导致空指针）
    data = [d for d in data if len(d.get("output", "")) > 10]
    print(f"Total valid samples after filtering short outputs: {len(data)}")

    # 3. 划分训练/验证集 (90% 训练, 10% 验证)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    # 💡 提示：因为 JSON 任务数据偏长，如果显存撑得住，可以把 max_length 256 改成 512
    train_dataset = InstructionDataset(train_data, tokenizer, max_length=256)
    val_dataset = InstructionDataset(val_data, tokenizer, max_length=256)

    collate = partial(collate_fn, pad_token_id=tokenizer.eot_token)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate)

    # 4. 初始化模型并加载【上一次微调】的权重
    model = GPTModel(
        vocab_size=cfg.vocab_size,
        embed_dim=cfg.embedding_dim,
        context_len=cfg.context_len,
        dropout=cfg.dropout,
        num_heads=cfg.num_heads,
        bias=cfg.bias,
        num_layers=cfg.num_layers
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"✅ 上一轮微调权重已成功加载自: {checkpoint_path}")

    # 5. 🎯 优化器参数微调：降低学习率（从 5e-5 降到 1e-5）
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    scheduler = warmup_scheduler(optimizer, warmup_steps=100)  # 数据量变小了，warmup步数也可以相应减少

    # 6. 记录loss用于可视化
    train_losses = []  # 每个epoch的平均train loss
    val_losses = []  # 每个epoch的平均val loss
    step_losses = []  # 每200步记录一次，画细粒度曲线

    # 7. 训练循环
    num_epochs = 4  # 增量微调不需要太多轮，3~4轮足够收敛
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for step, (input_ids, labels) in enumerate(train_loader):
            input_ids, labels = input_ids.to(device), labels.to(device)
            optimizer.zero_grad()

            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if step % 50 == 0:  # 总数据量变少后，调小打印间隔（从200改到50），方便肉眼观察
                current_lr = scheduler.get_last_lr()[0]
                step_losses.append(loss.item())  # 记录step级loss
                print(
                    f"Epoch {epoch + 1} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f} | LR: {current_lr:.2e}")

        # 验证损失
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input_ids, labels in val_loader:
                input_ids, labels = input_ids.to(device), labels.to(device)
                logits = model(input_ids)
                val_loss += torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100
                ).item()

        avg_train = total_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        print(f"⏰ [Epoch {epoch + 1} 结束] Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        # 每个epoch结束更新一次图
        plot_losses(train_losses, val_losses, step_losses, OUTPUT_DIR)

        # 保存 checkpoint
        os.makedirs(os.path.join(BASE_DIR, "checkpoints"), exist_ok=True)
        save_path = os.path.join(BASE_DIR, "checkpoints", f"json_finetune_major_epoch{epoch + 1}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"💾 新权重已存档: {save_path}")

    print("🎉 恭喜！针对JSON提取的增量指令微调已圆满完成!")
    print(f"最终 Loss 曲线图保存在: {OUTPUT_DIR}/instruct_loss_major_curve.png")


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        import traceback

        traceback.print_exc()
        input("按回车退出")  # 防止窗口闪退