import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import tiktoken
from config import cfg
from src.data.dataloader import create_dataloader
from src.models.gpt_model import GPTModel
from src.utils.training_utils import calc_loss_batch,calc_loss_loader
from src.utils.generation import generate_text_simple,text_to_ids,ids_to_text

if __name__=="__main__":
    # 加载数据
    file_path='./data/raw/the-verdict.txt'
    with open(file_path,'r',encoding='utf-8') as f:
        text=f.read()
    
    # 划分训练/验证集
    train_ratio=0.9
    split_idx=int(train_ratio*len(text))
    train_text=text[:split_idx]
    val_text=text[split_idx:]

    # 创建Dataloader
    tokenizer=tiktoken.get_encoding('gpt2')
    train_loader=create_dataloader(
        train_text,
        tokenizer,
        batch_size=2,
        max_len=256,
        stride=256,
        shuffle=True,
        drop_last=True,
    )
    val_loader=create_dataloader(
        val_text,
        tokenizer,
        batch_size=2,
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
    eval_freq=5 # 每几步评估一次
    eval_iter=3 # 评估时用5个batch
    start_context='Every effort moves you'

    train_losses=[]
    val_losses=[]
    tokens_seen=0
    global_step=-1

    for epoch in range(epochs):
        model.train()

        for input_batch,target_batch in train_loader:
            optimizer.zero_grad()
            loss=calc_loss_batch(input_batch,target_batch,model,device)
            loss.backward()
            optimizer.step()

            tokens_seen+=input_batch.numel()
            global_step+=1

            if global_step%eval_freq==0:
                train_loss=calc_loss_loader(train_loader,model,device,num_batches=eval_iter)
                val_loss=calc_loss_loader(val_loader,model,device,num_batches=eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                print(f"Epoch {epoch+1}/{epochs}, Step {global_step}",
                       f"Train loss {train_loss:.4f}, Val loss {val_loss:.4f}")
        
        model.eval()
        with torch.no_grad():
            encoded=text_to_ids(start_context,tokenizer).to(device)
            out=generate_text_simple(
                model,
                idx=encoded,
                max_new_tokens=50,
                context_size=cfg.context_len,
            )
            decoded=ids_to_text(out,tokenizer)
            print(f"\n生成样本 (Epoch {epoch+1}):\n{decoded}\n")
    
    torch.save(model.state_dict(),'./checkpoints/pretrained_model.pth')



