"""
Transformer Block
"""
import torch
import torch.nn as nn
from src.models.attention import Attention

class LayerNorm(nn.Module):
    """层归一化（带可学习的 scale 和 shift）"""
    def __init__(self,embed_dim):
        super().__init__()
        self.eps=1e-5
        self.scale=nn.Parameter(torch.ones(embed_dim))
        self.shift=nn.Parameter(torch.zeros(embed_dim))

    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        var=x.var(dim=-1,keepdim=True)
        out=(x-mean)/torch.sqrt(var+self.eps)
        out=self.scale*out+self.shift
        return out

# torch.manual_seed(123)
# batch_example=torch.randn(2,5)
# print(batch_example)
# layer=nn.Sequential(
#     nn.Linear(5,6),
#     nn.ReLU(),
# )
# out=layer(batch_example)
# ln=LayerNorm(6)
# out=ln(out)
# print(out)

class GELU(nn.Module):
    """GELU 激活函数"""
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        return 0.5*x*(1+torch.tanh(
            torch.sqrt(torch.tensor(2.0/torch.pi))*(x+0.044715*x**3)
        ))
    
class FeedForward(nn.Module):
    """前馈网络(FFN)"""
    def __init__(self,embed_dim):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Linear(embed_dim,embed_dim*4),
            GELU(),
            nn.Linear(embed_dim*4,embed_dim),
        )
    
    def forward(self,x):
        return self.layers(x)
    

class TransformerBlock(nn.Module):
    """一个完整的Transformer块"""
    def __init__(self,embed_dim,context_len,dropout,num_heads,bias):
        super().__init__()
        self.attn=Attention(
            d_in=embed_dim,
            d_out=embed_dim,
            context_len=context_len,
            dropout=dropout,
            num_heads=num_heads,
            bias=bias,
        )
        self.ff=FeedForward(embed_dim)
        self.norm1=LayerNorm(embed_dim)
        self.norm2=LayerNorm(embed_dim)
        self.dropout=nn.Dropout(dropout)
    
    def forward(self,x):
        # 第一个残差块
        shortcut=x
        x=self.norm1(x)
        x=self.attn(x)
        x=self.dropout(x)
        x=x+shortcut

        # 第二个残差块
        shortcut=x
        x=self.norm2(x)
        x=self.ff(x)
        x=self.dropout(x)
        x=x+shortcut

        return x
    

# x=torch.randn(2,5,32)
# print(x.shape)
# block=TransformerBlock(
#     embed_dim=32,
#     context_len=8,
#     dropout=0.1,
#     num_heads=4,
#     bias=False,
# )
# output=block(x)
# print(output.shape)




