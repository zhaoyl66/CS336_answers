import torch
import torch.nn as nn
from einops import rearrange, einsum
import math

class Linear(nn.Module):
    def __init__(self,in_features, out_features, device = None, dtype = None):
        """
        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(torch.empty(out_features, in_features,device=device, dtype= dtype))
        # pytorch是列向量表示，行向量表示的转置矩阵

        self._init_weight()
    
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        res = x @ self.weight.T
        # res = einsum(x,self.weight,"··· d_in, d_out d_in -> ··· d_out")
        return res

    # Xavier正态分布
    def _init_weight(self):
        std = math.sqrt(2 / (self.in_features + self.out_features)) #为了使y的方差与x的方差相等
        torch.nn.init.trunc_normal_(self.weight, mean = 0, std=std, a=-3*std, b=3*std)

class Embedding(nn.Module):
    def __init__(self,num_embeddings, embedding_dim, device=None,dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(torch.empty(num_embeddings,embedding_dim, device=device, dtype= dtype ))
        self._init_weight()

    def forward(self, token_ids:torch.Tensor)->torch.Tensor:
        if token_ids.dtype == torch.long:
            pass
        else:
            token_ids = token_ids.long()
        embeddings = self.weight[token_ids]
        return embeddings
    
    def _init_weight(self):
        nn.init.trunc_normal_(self.weight, std=1, a = -3, b=3)

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.g_weight = nn.Parameter(torch.empty(d_model,device=device,dtype= dtype))
        self._init_weight()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms_a = torch.sqrt(self.eps + torch.mean(x**2,dim=-1, keepdim=True))
        x_norm = x/rms_a
        return x_norm * self.g_weight.to(dtype=in_dtype)
    
    def _init_weight(self):
        nn.init.normal_(self.g_weight, std=1)
