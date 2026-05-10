"""
评估与测评
"""
import torch
import torch.nn.functional as F

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

