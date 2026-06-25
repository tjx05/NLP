import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import tiktoken

class ClsDataset(Dataset):

    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        # 要求 csv 包含 "Text" 和 "Label" 两列
        self.df = pd.read_csv(csv_file)
        
        print(f"正在对文本数据进行分词处理 ({csv_file})...")
        self.encoded_texts = [tokenizer.encode(str(text)) for text in self.df["Text"]]
        
        # 确定最大长度以进行截断或填充
        if max_length is None:
            self.max_length = max(len(t) for t in self.encoded_texts)
        else:
            self.max_length = max_length
            self.encoded_texts = [t[:self.max_length] for t in self.encoded_texts]
            
        # 使用 pad_token_id 填充至等长
        self.encoded_texts = [
            t + [pad_token_id] * (self.max_length - len(t)) 
            for t in self.encoded_texts
        ]

    def __getitem__(self, index):
        return (
            torch.tensor(self.encoded_texts[index], dtype=torch.long),
            torch.tensor(self.df.iloc[index]["Label"], dtype=torch.long)
        )

    def __len__(self):
        return len(self.df)

def get_cls_dataloaders(train_path, val_path, batch_size=8, max_length=None):
    """
    构建并返回训练与验证的 DataLoader
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    
    train_dataset = ClsDataset(train_path, tokenizer, max_length=max_length)
    # 保证验证集的最大长度和训练集一致
    val_dataset = ClsDataset(val_path, tokenizer, max_length=train_dataset.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, val_loader, train_dataset.max_length