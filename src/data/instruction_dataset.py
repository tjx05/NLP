# src/data/instruction_dataset.py
import json
import torch
from torch.utils.data import Dataset


class InstructionDataset(Dataset):
    """
    处理 Alpaca 格式的指令数据:
    {"instruction": "...", "input": "...", "output": "..."}

    格式化为:
    ### Instruction:
    {instruction}

    ### Input:
    {input}  # 可为空

    ### Response:
    {output}
    """

    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encoded = [self._encode(item) for item in data]

    def _format_prompt(self, item):
        if item.get("input", "").strip():
            return (
                f"### Instruction:\n{item['instruction']}\n\n"
                f"### Input:\n{item['input']}\n\n"
                f"### Response:\n"
            )
        return (
            f"### Instruction:\n{item['instruction']}\n\n"
            f"### Response:\n"
        )

    def _encode(self, item):
        prompt = self._format_prompt(item)
        full_text = prompt + item["output"]

        prompt_ids = self.tokenizer.encode(prompt)
        full_ids = self.tokenizer.encode(full_text)

        # 截断
        full_ids = full_ids[:self.max_length]

        input_ids = torch.tensor(full_ids[:-1], dtype=torch.long)
        labels = torch.tensor(full_ids[1:], dtype=torch.long)

        # 只对 response 部分计算 loss，prompt 部分 mask 掉
        prompt_len = len(prompt_ids)
        labels[:prompt_len - 1] = -100  # -100 会被 CrossEntropyLoss 忽略

        return input_ids, labels

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        return self.encoded[idx]


def collate_fn(batch, pad_token_id=50256):
    """动态 padding"""
    input_ids, labels = zip(*batch)
    max_len = max(x.shape[0] for x in input_ids)

    padded_inputs = torch.full((len(input_ids), max_len), pad_token_id, dtype=torch.long)
    padded_labels = torch.full((len(labels), max_len), -100, dtype=torch.long)

    for i, (inp, lbl) in enumerate(zip(input_ids, labels)):
        padded_inputs[i, :inp.shape[0]] = inp
        padded_labels[i, :lbl.shape[0]] = lbl

    return padded_inputs, padded_labels