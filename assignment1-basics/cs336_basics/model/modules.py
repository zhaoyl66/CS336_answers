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

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        if dtype is None or not torch.is_floating_point(torch.empty((),dtype=dtype)):
            dtype = torch.float32
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        # self.w1 = nn.Parameter(torch.empty(d_ff,d_model,device = device, dtype = dtype))
        # self.w2 = nn.Parameter(torch.empty(d_model,d_ff,device = device, dtype = dtype))
        # self.w3 = nn.Parameter(torch.empty(d_ff,d_model,device = device, dtype = dtype))
        self.w1 = Linear(d_model,d_ff,device = device, dtype = dtype)
        self.w2 = Linear(d_ff,d_model,device = device, dtype = dtype)
        self.w3 = Linear(d_model,d_ff,device = device, dtype = dtype)
        self._init_weight()
    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        a = self.w1(x)
        silu = a * torch.sigmoid(a)
        b = self.w3(x)
        c = silu * b
        d = self.w2(c)
        return d


    def _init_weight(self):
        std = (2/(self.d_ff + self.d_model)) ** 0.5
        nn.init.trunc_normal_(self.w1.weight,std = std, a = -3*std, b = 3*std)
        nn.init.trunc_normal_(self.w2.weight,std = std, a = -3*std, b = 3*std)
        nn.init.trunc_normal_(self.w3.weight,std = std, a = -3*std, b = 3*std)



class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.freq_arrange = 1 / (self.theta**(torch.arange(0, d_k, 2).to(dtype=torch.float)/self.d_k))
        #  inv_freq： $$\text{freq}_i = \frac{1}{\theta^{2i/d_k}}, \quad i = 0, 1, 2, ..., \frac{d_k}{2}-1$$
        self.register_buffer(name='inv_freq',tensor=self.freq_arrange)
        # buffer fixed，no update

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        """
        input/output tensor shape (..., seq_len, d_k):  (Batch_size, hidden_size,seq_len,d_k) or (Batch_size,seq_len,d_k)
        token_positions: (..., seq_len)
        """
        S = x.size(-2)  # seq_len

        #  token_positions
        if token_positions is None:
            token_positions = torch.arange(S, device=x.device, dtype=torch.float)
        else:
            if token_positions.dim() == 2:
                token_positions = token_positions[0]
            token_positions = token_positions.to(dtype=torch.float)
        # make sure token_positions (S,)

        # (S, d_k/2) position
        theta = torch.outer(token_positions, self.inv_freq)

        # 计算 cos 和 sin: (S, d_k/2)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        # 扩展到完整维度: (S, d_k/2) -> (S, d_k)
        cos_theta = torch.repeat_interleave(cos_theta, 2, dim=-1)  # (S, d_k)
        sin_theta = torch.repeat_interleave(sin_theta, 2, dim=-1)  # (S, d_k)

        # 将 x 分解为两部分用于旋转
        # x1 = [x0, x2, x4, ...], x2 = [x1, x3, x5, ...]
        x1 = x[..., ::2]  # 偶数索引: (..., S, d_k/2)
        x2 = x[..., 1::2]  # 奇数索引: (..., S, d_k/2)

        # 构造旋转后的向量
        # x'_even = x_even * cos - x_odd * sin
        # x'_odd = x_even * sin + x_odd * cos
        x_rotated_even = x1 * cos_theta[..., ::2] - x2 * sin_theta[..., ::2]
        x_rotated_odd = x1 * sin_theta[..., 1::2] + x2 * cos_theta[..., 1::2]

        # 交错合并回原始形状: (..., S, d_k)
        x_out = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
        x_out = x_out.flatten(start_dim=-2)  # (..., S, d_k)

        return x_out

def softmax(x: torch.Tensor, dim: int, temperature: float = 1.0) -> torch.Tensor:
    """
    Numerically stable softmax implementation with temperature scaling.

    softmax(x_i) = exp(x_i / τ) / sum(exp(x_j / τ))

    Args:
        x: Input tensor
        dim: Dimension along which to apply softmax
        temperature: Temperature parameter τ (default=1.0)
                    - τ < 1: sharper distribution (amplify differences)
                    - τ = 1: standard softmax
                    - τ > 1: smoother distribution (reduce differences)
    """
    # Apply temperature scaling
    x_scaled = x / temperature

    # Subtract max for numerical stability
    x_shifted = x_scaled - torch.max(x_scaled, dim=dim, keepdim=True).values

    exp_x = torch.exp(x_shifted)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Scaled Dot-Product Attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        q: Query tensor (B, S_q, D)
        k: Key tensor (B, S_k, D)
        v: Value tensor (B, S_v, D)
        mask: Optional mask (B, S_q, S_k) - False positions will be masked out

    Returns:
        Attention output (B, S_q, D)
    """
    d_k = q.size(-1) ** 0.5

    # Step 1: Compute QK^T
    # q: (..., S_q, D), k.T: (..., D, S_k) -> result: (..., S_q, S_k)
    q_k_score = q @ k.transpose(-2, -1)

    # Step 2: Scale by sqrt(d_k)
    attn_score = q_k_score / d_k

    # Step 3: Apply mask (optional)
    if mask is not None:
        attn_score = attn_score.masked_fill(mask == False, float('-inf'))

    # Step 4: Apply softmax to get attention weights
    attn_weights = softmax(attn_score, dim=-1)

    # Step 5: Multiply by V
    return attn_weights @ v
        
