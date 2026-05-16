"""
数据加载器
数据集划分+分布式训练
"""
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class GPTDataset(Dataset):
    def __init__(self, bin_file, max_len, split="train", stride=None):
        full_data=np.memmap(bin_file, dtype=np.uint16, mode='r')
        
        split_idx=int(0.9 * len(full_data))
        if split=="train":
            self.data=full_data[:split_idx]
        else:
            self.data=full_data[split_idx:]
            
        self.max_len=max_len
        # stride 默认等于 max_len，即不重叠切分（最常用）
        self.stride=stride if stride is not None else max_len

    def __len__(self):
        # 非重叠切分后的样本数，远小于滑窗方式
        return (len(self.data)-self.max_len)//self.stride

    def __getitem__(self, idx):
        # 按 stride 定位起始位置
        start=idx*self.stride
        d = self.data[start : start + self.max_len + 1].astype(np.int64)
        t = torch.from_numpy(d)
        x = t[:-1]
        y = t[1:]
        return x, y

def create_dataloader(bin_file, split="train", batch_size=32, max_len=1024,
                      shuffle=True, pin_memory=True, is_distributed=False, stride=None):
    dataset = GPTDataset(bin_file, max_len, split=split, stride=stride)
    
    # 如果是分布式训练，使用分布式采样器
    sampler = DistributedSampler(dataset, shuffle=shuffle) if is_distributed else None
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False if is_distributed else shuffle, # 如果用了 sampler，必须把 Dataloader 的 shuffle 设为 False
        sampler=sampler,
        pin_memory=pin_memory,
        num_workers=0,
        drop_last=True
    )
    return dataloader

# dataloader=create_dataloader(
#     raw_text,
#     batch_size=2,
#     max_len=4,
#     stride=1,
#     shuffle=False,
# )
# print("\n第一个 batch（输入 vs 目标）：")
# for idx, (input_ids, target_ids) in enumerate(dataloader):
#     print(f"输入: {input_ids}")
#     print(f"目标: {target_ids}")
#     break  # 只看第一个 batch