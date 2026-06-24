# scripts/run_json_finetune.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import random
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from functools import partial

import tiktoken
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from config import cfg
from src.models.gpt_model import GPTModel
from src.data.instruction_dataset import InstructionDataset, collate_fn

BASE_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR    = os.path.join(BASE_DIR, "output")

# ── 起点权重：在 instruct_epoch1.pt 基础上继续微调 ──
START_CKPT    = os.path.join(BASE_DIR, "checkpoints", "instruct_epoch1.pt")

# ── 三个数据集路径 ──
PATH_DOMAIN   = os.path.join(BASE_DIR, "data", "instrc_raw", "sft_LLMgenerate_dataset.json")
PATH_NEWS     = os.path.join(BASE_DIR, "data", "instrc_raw", "instruct_1_top100_processed.json")
PATH_OLD      = os.path.join(BASE_DIR, "data", "raw",        "old_instruct.json")


# ── 数据加载 ──────────────────────────────────────────────
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_all_data():
    domain = load_json(PATH_DOMAIN)   # ~700条 金融/医疗/科技
    news   = load_json(PATH_NEWS)     # ~100条 新闻/人名/日期提取
    old    = load_json(PATH_OLD)      # ~100条 旧指令，防遗忘

    print(f"  专业领域数据: {len(domain)} 条")
    print(f"  日常新闻数据: {len(news)} 条")
    print(f"  旧指令数据:   {len(old)} 条")

    combined = domain + news + old
    random.shuffle(combined)
    print(f"  合并后总计:   {len(combined)} 条")
    return combined


# ── Warmup ────────────────────────────────────────────────
def warmup_scheduler(optimizer, warmup_steps):
    def lr_lambda(step):
        return step / warmup_steps if step < warmup_steps else 1.0
    return LambdaLR(optimizer, lr_lambda)


# ── 可视化 ────────────────────────────────────────────────
def plot_losses(train_losses, val_losses, step_losses, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(step_losses, color='steelblue', linewidth=1.2, alpha=0.8)
    ax1.set_title("Training Loss (per 50 steps)", fontsize=13)
    ax1.set_xlabel("Record Index")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle='--', alpha=0.5)

    epochs = range(1, len(train_losses) + 1)
    ax2.plot(epochs, train_losses, marker='o', label='Train Loss', color='steelblue')
    ax2.plot(epochs, val_losses,   marker='s', label='Val Loss',   color='tomato')
    ax2.set_title("Train vs Val Loss (per epoch)", fontsize=13)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "json_finetune_loss_curve_epc10.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Loss curve saved: {save_path}")


# ── 训练 ──────────────────────────────────────────────────
def train():
    device    = cfg.device
    print(f"Using device: {device}")
    tokenizer = tiktoken.get_encoding("gpt2")

    # 1. 加载数据
    print("\n加载数据集...")
    data = load_all_data()
    # 过滤低质量，按难度排序
    data = [d for d in data if len(d.get("output", "")) > 10]
    data = sorted(data, key=lambda x: len(x["output"]))
    print(f"过滤后总计: {len(data)} 条")

    # 2. 划分训练/验证集
    split      = int(0.9 * len(data))
    train_data = data[:split]
    val_data   = data[split:]

    train_dataset = InstructionDataset(train_data, tokenizer, max_length=512)
    val_dataset   = InstructionDataset(val_data,   tokenizer, max_length=512)

    collate = partial(collate_fn, pad_token_id=tokenizer.eot_token)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(val_dataset,   batch_size=4, shuffle=False, collate_fn=collate)

    # 3. 加载模型，从 instruct_epoch1.pt 继续微调
    print(f"\n加载起点权重: {START_CKPT}")
    model = GPTModel(
        vocab_size=cfg.vocab_size,
        embed_dim=cfg.embedding_dim,
        context_len=cfg.context_len,
        dropout=cfg.dropout,
        num_heads=cfg.num_heads,
        bias=cfg.bias,
        num_layers=cfg.num_layers
    ).to(device)
    model.load_state_dict(torch.load(START_CKPT, map_location=device, weights_only=True))
    print("权重加载完毕，开始继续微调...")

    # 4. 优化器：lr 比上一阶段再小一点，避免覆盖已学知识
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = warmup_scheduler(optimizer, warmup_steps=100)

    # 5. 训练循环
    train_losses, val_losses, step_losses = [], [], []
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for step, (input_ids, labels) in enumerate(train_loader):
            input_ids, labels = input_ids.to(device), labels.to(device)
            optimizer.zero_grad()

            logits = model(input_ids)
            loss   = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if step % 50 == 0:
                current_lr = scheduler.get_last_lr()[0]
                step_losses.append(loss.item())
                print(f"Epoch {epoch+1} | Step {step} | Loss: {loss.item():.4f} | LR: {current_lr:.2e}")

        # 验证
        # 验证损失
        model.eval()
        val_loss = 0
        val_count = 0
        with torch.no_grad():
            for input_ids, labels in val_loader:
                input_ids, labels = input_ids.to(device), labels.to(device)
                # 跳过全是 -100 的 batch
                if (labels != -100).sum() == 0:
                    continue
                logits = model(input_ids)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100
                )
                if not torch.isnan(loss):
                    val_loss += loss.item()
                    val_count += 1

        avg_val = val_loss / val_count if val_count > 0 else float('nan')

        avg_train = total_loss / len(train_loader)
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        plot_losses(train_losses, val_losses, step_losses, OUTPUT_DIR)

        # 保存，用 json_ 前缀区分
        os.makedirs(os.path.join(BASE_DIR, "checkpoints"), exist_ok=True)
        save_path = os.path.join(BASE_DIR, "checkpoints", f"egin_json_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"Saved: {save_path}")

    print("\nJSON 专项微调完成！")


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        import traceback
        traceback.print_exc()
        input("按回车退出")