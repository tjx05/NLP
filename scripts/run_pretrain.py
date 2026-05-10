import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import tiktoken
from tqdm import tqdm
from config import cfg
from src.data.dataloader import create_dataloader
from src.models.gpt_model import GPTModel
from src.utils.evaluation import calc_loss_batch,calc_loss_loader
from src.utils.generation import generate_text_simple,generate_temp_topk,text_to_ids,ids_to_text

if __name__=="__main__":
    # 加载数据
    # file_path='./data/raw/mixed_pretrain_test.txt'
    # with open(file_path,'r',encoding='utf-8') as f:
    #     text=f.read()
    file_path='./data/processed/pretrain_tokens.pt'
    token_tensor=torch.load(file_path)
    raw_token_ids=token_tensor.tolist()
    
    # 划分训练/验证集
    train_ratio=0.9
    split_idx=int(train_ratio*len(raw_token_ids))
    train_ids=raw_token_ids[:split_idx]
    val_ids=raw_token_ids[split_idx:]

    # 创建Dataloader
    tokenizer=tiktoken.get_encoding('gpt2')
    train_loader=create_dataloader(
        train_ids,
        batch_size=2,
        max_len=256,
        stride=256,
        shuffle=True,
        drop_last=True,
    )
    val_loader=create_dataloader(
        val_ids,
        batch_size=8,
        max_len=256,
        stride=256,
        shuffle=False,
        drop_last=False,
    )
    
    # 创建模型
    device=cfg.device
    model=GPTModel(
        vocab_size=cfg.vocab_size,
        embed_dim=cfg.embedding_dim,
        context_len=cfg.context_len,
        dropout=cfg.dropout,
        num_heads=cfg.num_heads,
        bias=cfg.bias,
        num_layers=cfg.num_layers,
    )
    model=model.to(device)
    
    # 优化器
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.lr)

    # 训练
    epochs=cfg.epochs
    start_context='Every effort moves you'
    patience=5  # 早停耐心轮数
    min_delta = 0.001     # 只有下降超过这个值才算改善
    counter=0

    best_val_loss=float('inf')

    print(f"开始训练，总计 {epochs} 个 epoch")
    print(f"数据: {len(train_ids):,} tokens, {len(train_loader)} batches\n")
    for epoch in range(epochs):
        model.train()
        epoch_loss=0.0
        num_batches=0

        # 进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        for input_batch,target_batch in pbar:
            optimizer.zero_grad()
            loss=calc_loss_batch(input_batch,target_batch,model,device)
            loss.backward()
            optimizer.step()

            epoch_loss+=loss.item()
            num_batches+=1
            pbar.set_postfix({"loss":f"{epoch_loss/num_batches:.4f}"})
        
        model.eval()
        with torch.no_grad():
            # 验证损失
            val_loss=calc_loss_loader(val_loader,model,device)
            print(f"Epoch {epoch+1}/{epochs}, Val loss {val_loss:.4f}")

            # 保存模型
            if val_loss<best_val_loss-min_delta:
                best_val_loss=val_loss
                counter=0
                torch.save(model.state_dict(),'./checkpoints/best_pretrained_model.pth')
                print(f"保存新最佳模型，验证损失: {best_val_loss:.4f}")
            else:
                counter+=1

            # 早停判断
            if counter>=patience:
                print(f"早停，验证损失未改进{patience}个epoch")
                break
            
            encoded=text_to_ids(start_context,tokenizer).to(device)
            # out=generate_text_simple(
            #     model,
            #     idx=encoded,
            #     max_new_tokens=50,
            #     context_size=cfg.context_len,
            # )
            out=generate_temp_topk(
                model,
                idx=encoded,
                max_new_tokens=50,
                context_size=cfg.context_len,
                temp=1.0,
                top_k=50,
                # eos_id=tokenizer.eos_token_id,
            )
            decoded=ids_to_text(out,tokenizer)
            print(f"\n生成样本 (Epoch {epoch+1}):\n{decoded}\n")

        # 保存模型(用于恢复训练）)
        torch.save(model.state_dict(),f'./checkpoints/epoch_{epoch+1}.pth')



