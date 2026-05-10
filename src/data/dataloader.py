"""
数据加载器
"""
import tiktoken
import torch
from torch.utils.data import Dataset,DataLoader

class GPTDataset(Dataset):
    """直接接收已分词好的 token ids"""
    def __init__(self,token_ids,max_len,stride):
        self.input_ids=[]
        self.target_ids=[]
        # token_ids=tokenizer.encode(txt)

        for i in range(0,len(token_ids)-max_len,stride):
            input_chunk=token_ids[i:i+max_len]
            target_chunk=token_ids[i+1:i+max_len+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self,idx):
        return self.input_ids[idx],self.target_ids[idx]

def create_dataloader(token_ids,batch_size=4,max_len=256,stride=128,shuffle=True,drop_last=True):
    dataset=GPTDataset(token_ids,max_len,stride)
    dataloader=DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
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