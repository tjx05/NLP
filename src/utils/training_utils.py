# bibibiibii
import torch
import torch.nn.functional as F
from src.utils.generation import generate_text_simple,text_to_ids,ids_to_text

def calc_loss_batch(input_batch, target_batch, model, device):
    """计算单个batch的损失"""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    
    logits = model(input_batch)  # [batch, seq_len, vocab_size]
    
    # 交叉熵损失需要展平
    loss = F.cross_entropy(
        logits.flatten(0, 1),  # [batch*seq_len, vocab_size]
        target_batch.flatten() # [batch*seq_len]
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """计算整个DataLoader的平均损失"""
    total_loss = 0.
    
    if len(data_loader) == 0:
        return float("nan")
    
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """评估训练集和验证集损失"""
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    """生成并打印一个样本（用于观察训练进度）"""
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    
    # 编码输入
    encoded = text_to_ids(start_context, tokenizer).to(device)
    
    # 生成
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size
        )
    
    # 解码并打印
    decoded_text = ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # 去掉换行，方便显示
    
    model.train()