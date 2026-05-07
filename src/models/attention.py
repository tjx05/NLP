"""
注意力机制
"""
import torch
import torch.nn as nn

class Attention(nn.Module):
    """
    多头注意力机制
    """
    def __init__(self,d_in,d_out,context_len,dropout,num_heads,bias=False):
        super().__init__()
        self.num_heads=num_heads
        self.head_dim=d_out//num_heads
        self.d_out=d_out

        # 权重矩阵
        self.w_q=nn.Linear(d_in,d_out,bias=bias)
        self.w_k=nn.Linear(d_in,d_out,bias=bias)
        self.w_v=nn.Linear(d_in,d_out,bias=bias)
        self.out_proj=nn.Linear(d_out,d_out,bias=bias) # 输出投影
        self.dropout=nn.Dropout(dropout)

        # 掩码,注册因果mask（不参与训练，自动跟随设备）
        self.register_buffer("mask",torch.triu(torch.ones(context_len,context_len),diagonal=1)) # 不是模型参数，不训练


    def forward(self,x):
        b,num_tokens,d_in=x.shape

        # 拆分成多头的形状 
        queries=self.w_q(x).view(b,num_tokens,self.num_heads,self.head_dim)
        keys=self.w_k(x).view(b,num_tokens,self.num_heads,self.head_dim)
        values=self.w_v(x).view(b,num_tokens,self.num_heads,self.head_dim)

        # (b,num_tokens,num_heads,head_dim)-->(b,num_heads,num_tokens,head_dim)
        queries=queries.transpose(1,2)
        keys=keys.transpose(1,2)
        values=values.transpose(1,2)

        # 计算注意力分数
        attn_scores=queries@keys.transpose(-2,-1)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens,:num_tokens],-torch.inf)
        # 缩放+softmax
        d_k=keys.shape[-1]
        attn_weights=torch.softmax(attn_scores/d_k**0.5,dim=-1)
        attn_weights=self.dropout(attn_weights)

        # 计算上下文向量
        context_vec=attn_weights@values

        # 合并多头的上下文向量
        context_vec=context_vec.transpose(1,2)
        context_vec=context_vec.reshape(b,num_tokens,self.d_out)
        # 输出投影
        context_vec=self.out_proj(context_vec)


        return context_vec

# 测试多头注意力
# if __name__ == "__main__":
#     batch = 2
#     seq = 8
#     d_in = 16
#     d_out = 32
#     num_heads = 4
    
#     x = torch.randn(batch, seq, d_in)
#     mha = Attention(d_in, d_out, context_len=seq, dropout=0.1, num_heads=num_heads)
#     out = mha(x)
    
#     print(f"多头注意力输出 shape: {out.shape}")  # 期望 [2, 8, 32]
    


