import json
import matplotlib.pyplot as plt
import os

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import tiktoken
from tqdm import tqdm
import json
import math
import os

# 导入你自己的模块
from config import cfg
from src.data.dataloader import create_dataloader
from src.models.gpt_model import GPTModel
from src.utils.evaluation import calc_loss_batch, calc_loss_loader
from src.utils.generation import generate_temp_topk, text_to_ids, ids_to_text

if __name__=="__main__":
    # ---------------------------------------------------------
    #初始化 DDP (分布式通信)
    # ---------------------------------------------------------
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    master_process = local_rank == 0

    if master_process:
        os.makedirs("logs", exist_ok=True)
        os.makedirs("logs/samples", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

    tokenizer = tiktoken.get_encoding('gpt2')
    
    # ---------------------------------------------------------
    # 挂载数据集
    # ---------------------------------------------------------
    if master_process: print("正在挂载数据集 (分布式 Memmap 模式)...")
    bin_file_path = './data/processed/train.bin'
    
    train_loader = create_dataloader(
        bin_file_path, split="train", batch_size=cfg.batch_size, max_len=cfg.context_len, shuffle=True, is_distributed=True
    )
    val_loader = create_dataloader(
        bin_file_path, split="val", batch_size=cfg.batch_size, max_len=cfg.context_len, shuffle=False, is_distributed=True
    )

    # ---------------------------------------------------------
    # 创建模型并使用 DDP 包裹
    # ---------------------------------------------------------
    model = GPTModel(
        vocab_size=cfg.vocab_size,
        embed_dim=cfg.embedding_dim,
        context_len=cfg.context_len,
        dropout=cfg.dropout,
        num_heads=cfg.num_heads,
        bias=cfg.bias,
        num_layers=cfg.num_layers,
    ).to(device)
    
    # 续训：加载第1轮权重
    checkpoint_path = "./checkpoints/epoch_1.pth"
    if os.path.exists(checkpoint_path):
        if master_process:
            print(f"加载第1轮模型，从第2轮开始训练")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model = DDP(model, device_ids=[local_rank])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    # ---------------------------------------------------------
    # cosine decay with warmup
    # ---------------------------------------------------------
    total_steps = cfg.epochs * len(train_loader)

    def get_lr(step):
        if step < cfg.warmup_steps:
            return cfg.lr * step / cfg.warmup_steps
        ratio = (step - cfg.warmup_steps) / (total_steps - cfg.warmup_steps)
        return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * ratio))

    history = {
        "epochs": [], "train_loss": [], "val_loss": [], "train_ppl": [], "val_ppl": [],
        "step_loss": [], "step_ppl": [], "step_lr": []  # 去掉steps，加上step_lr
    }
    start_context = 'Every effort moves you'

    if master_process:
        print(f"\n 开始 4 卡并行预训练！总计 {cfg.epochs} 个 epoch")
        print(f" 每张卡分配的 Batch 数量预估: {len(train_loader):,}\n")

    # ---------------------------------------------------------
    # 训练主循环
    # ---------------------------------------------------------
    global_step = 0  # 跨epoch的全局步数，用于lr调度

    for epoch in range(cfg.epochs):
        train_loader.sampler.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}", unit="batch", disable=not master_process)
        
        for input_batch, target_batch in pbar:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            # 更新学习率
            lr = get_lr(global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = calc_loss_batch(input_batch, target_batch, model, device)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            # 每10000步记录一次
            if master_process and global_step % 1000 == 0:
                history["step_loss"].append(loss.item())
                history["step_ppl"].append(math.exp(loss.item()))
                history["step_lr"].append(lr)
                with open("logs/history.json", "w") as f:
                    json.dump(history, f, indent=2)

            if master_process and num_batches % 10 == 0:
                pbar.set_postfix({"loss": f"{epoch_loss/num_batches:.4f}", "lr": f"{lr:.2e}"})
            
            # 阶段性存档
            if master_process and num_batches % 5000 == 0:
                torch.save(model.module.state_dict(), f'./checkpoints/step_{num_batches}.pth')
                print(f"\n[存档] 已完成 {num_batches} 步，阶段模型已保存。")
        
        avg_train_loss = epoch_loss / num_batches
        
        # ---------------------------------------------------------
        # 验证与生成测试
        # ---------------------------------------------------------
        model.eval()
        with torch.no_grad():
            val_loss = calc_loss_loader(val_loader, model.module, device, num_batches=200)
            
            if master_process:
                print(f"Epoch {epoch+1}/{cfg.epochs}, Val loss {val_loss:.4f}")
                history["epochs"].append(epoch + 1)
                history["train_loss"].append(avg_train_loss)
                history["val_loss"].append(val_loss)
                history["train_ppl"].append(math.exp(avg_train_loss))
                history["val_ppl"].append(math.exp(val_loss))

                torch.save(model.module.state_dict(), f'./checkpoints/epoch_{epoch+1}.pth')

                best_model_path = "./checkpoints/best.pth"
                best_val_loss = min(history["val_loss"])
                if not os.path.exists(best_model_path) or val_loss <= best_val_loss:
                    torch.save(model.module.state_dict(), best_model_path)
                    print(f"已更新【最佳模型】best.pth | val_loss = {val_loss:.4f}")

                encoded = text_to_ids(start_context, tokenizer).to(device)
                out = generate_temp_topk(model.module, idx=encoded, max_new_tokens=50, context_size=cfg.context_len, temp=1.0, top_k=50)
                decoded = ids_to_text(out, tokenizer)
                print(f"\n 生成样本 (Epoch {epoch+1}):\n{decoded}\n")

                with open(f"logs/samples/epoch_{epoch+1}.txt", "w", encoding="utf-8") as f:
                    f.write(f"Epoch: {epoch+1}\n")
                    f.write(f"Prompt: {start_context}\n\n")
                    f.write(f"Generated:\n{decoded}\n")

                with open("logs/history.json", "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2)
                print("\n训练历史已保存到 logs/history.json")
    
    dist.destroy_process_group()