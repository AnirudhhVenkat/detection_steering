import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class Config:
  d_model: int = 384
  n_heads: int = 6
  n_layers: int = 6
  ff_d_ratio: int=4


class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        #get config params
        self.config = config
        self.d_model = self.config.d_model
        self.n_heads = self.config.n_heads
        self.n_layers = self.config.n_layers
        self.ff_d_ratio = self.config.ff_d_ratio

        
        #model layers
        self.qkv = nn.Linear(self.d_model, 3*self.d_model)
        self.out = nn.Linear(self.d_model, self.d_model)
    
    def forward(self, x):
        B,T,D = x.shape 
        qkv = self.qkv(x)
        q,k,v = torch.split(x, split_size_or_sections=3)

        head_dim = self.d_model // self.n_heads

        q,k,v = q.reshape(B,T,self.n_heads,head_dim), k.reshape(B,T,self.n_heads,head_dim), v.reshape(B,T,self.n_heads,head_dim)

        q,k,v = torch.tranpose(q,1,2), torch.tranpose(k,1,2), torch.tranpose(v,1,2)

        att = q @ torch.tranpose(k,2,3)

        att = att/math.sqrt(head_dim)

        mask = torch.tril(torch.ones((T,T)))

        att = torch.where(mask[None,None,:,:],att,-1e9)

        att = F.softmax(att, axis=-1)

        out = att @ v

        out = torch.transpose(out,1,2)

        out = out.reshape(B,T,D)

        out = self.out(out)

        return out 


class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.d_model = self.config.d_model

        self.down_proj = nn.Linear(self.d_model, 4*self.d_model)
        self.out_proj = nn.Linear(4*self.d_model, self.d_model)
    
    def forward(self, x):
        x = self.down_proj(x)
        x = F.gelu(x)
        x = self.out_proj(x)

        return x


class Transformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.attention = Attention(config)
        self.mlp = MLP(config)
        self.norm = nn.LayerNorm(self.config.d_model)
    def forward(self, x):
        x = x + self.attention(self.norm(x))
        x = x + self.mlp(self.norm(x))
        return x




















