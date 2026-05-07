"""
GPT模型集合
"""
import torch
import torch.nn as nn
from src.models.transformer_block import TransformerBlock,LayerNorm 

class GPTModel(nn.Module):
    def __init__(self,vocab_size,embed_dim,context_len,dropout,num_heads,bias,num_layers):
        super().__init__()
        # 词嵌入+位置编码
        self.token_embed=nn.Embedding(vocab_size,embed_dim)
        self.pos_embed=nn.Embedding(context_len,embed_dim)
        self.dropout=nn.Dropout(dropout)

        # 多个Transformer Block堆叠
        self.trf_blocks=nn.Sequential(
            *[TransformerBlock(embed_dim,context_len,dropout,num_heads,bias) for _ in range(num_layers)]
        )

        # 最终层归一化
        self.final_norm=LayerNorm(embed_dim)

        # 输出层
        self.output=nn.Linear(embed_dim,vocab_size,bias=False)

    def forward(self,input_ids):
        """input_ids: [batch_size,seq_len]"""
        batch_size,seq_len=input_ids.shape

        # 词嵌入+位置编码
        token_embeds=self.token_embed(input_ids)
        pos_embeds=self.pos_embed(torch.arange(seq_len).to(input_ids.device))
        x=token_embeds+pos_embeds
        x=self.dropout(x)

        # 通过所有Transformer Block
        x=self.trf_blocks(x)

        # 最终归一化
        x=self.final_norm(x)

        # 输出层
        logits=self.output(x) # [batch, seq_len, vocab_size]

        return logits
    
# emb_dim = 32      # 临时缩小
# n_heads = 4       # 临时缩小
# n_layers = 2      # 只堆2层，快速测试
# context_length = 8
# vocab_size = 100   # 临时缩小词表

# batch_size = 2
# seq_len = 5
# input_ids = torch.randint(0,vocab_size,(batch_size,seq_len))
# model=GPTModel(vocab_size,emb_dim,context_length,0.1,n_heads,False,n_layers)
# output=model(input_ids)
# print(output.shape)

# total_params = sum(p.numel() for p in model.parameters())
# print(f"\n模型总参数量（临时配置）: {total_params:,}")






