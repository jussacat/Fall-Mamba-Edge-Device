import torch
import torch.nn as nn
import math
from typing import Optional
from torch import Tensor
from functools import partial
from timm.models.layers import DropPath
import torch.utils.checkpoint as checkpoint
from .mamba_minimal import MambaBlock, ModelArgs, RMSNorm

class MultiHeadTemporalAttention(nn.Module):
    def __init__(self, in_dim, num_heads=8, dropout_rate=0.1, max_len=5000):
        super().__init__()
        assert in_dim % num_heads == 0, "in_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.depth = in_dim // num_heads

        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_rate)
        self.scale = nn.Parameter(torch.sqrt(torch.tensor(self.depth, dtype=torch.float32)))
        self.norm = nn.LayerNorm(in_dim)

        nn.init.kaiming_uniform_(self.query.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.key.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.value.weight, mode='fan_in', nonlinearity='relu')

        self.time_pos_encoding = self._generate_time_pos_encoding(max_len, in_dim)

    def _generate_time_pos_encoding(self, max_len, d_model):
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pos_encoding = torch.zeros(max_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        pos_encoding = pos_encoding.unsqueeze(0)
        return pos_encoding

    def forward(self, x, seq_len=None):
        batch_size, seq_len, _ = x.size()

        time_pos_encoding = self.time_pos_encoding[:, :seq_len, :].to(x.device)
        x = x + time_pos_encoding

        query = self.query(x).view(batch_size, seq_len, self.num_heads, self.depth).transpose(1, 2)
        key = self.key(x).view(batch_size, seq_len, self.num_heads, self.depth).transpose(1, 2)
        value = self.value(x).view(batch_size, seq_len, self.num_heads, self.depth).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        attn = self.softmax(scores)
        attn = self.dropout(attn)

        out = torch.matmul(attn, value).transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.depth)
        out = self.norm(out)
        return out

class Block(nn.Module):
    def __init__(self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False,
                 residual_in_fp32=False, drop_path=0., apply_temporal_attention=False,
                 num_heads=8, mlp_ratio=4.0, max_len=5000):
        super().__init__()
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32
        self.apply_temporal_attention = apply_temporal_attention
        if apply_temporal_attention:
            self.temporal_attention = MultiHeadTemporalAttention(dim, num_heads=num_heads, max_len=max_len)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(mlp_hidden_dim, dim),
                nn.Dropout(drop_path)
            )

    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None,
                use_checkpoint=False):
        if residual is None:
            residual = hidden_states

        if self.apply_temporal_attention:
            temp_output = self.temporal_attention(hidden_states)
            hidden_states = temp_output + residual
            hidden_states = self.mlp(hidden_states)

        if not self.fused_add_norm:
            residual = (residual + self.drop_path(hidden_states)) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)

        if use_checkpoint:
            hidden_states = checkpoint.checkpoint(self.mixer, hidden_states)
        else:
            hidden_states = self.mixer(hidden_states)

        if not self.apply_temporal_attention:
            if not self.fused_add_norm:
                residual = (residual + self.drop_path(hidden_states)) if residual is not None else hidden_states
                hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states if residual is None else self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

def create_block(d_model, ssm_cfg=None, norm_epsilon=1e-5, drop_path=0., 
                 rms_norm=True, residual_in_fp32=True, fused_add_norm=True, 
                 layer_idx=None, bimamba=True, device=None, dtype=None, 
                 apply_temporal_attention=False, num_heads=8, max_len=5000):
    
    if ssm_cfg is None:
        ssm_cfg = {}
        
    # Cấu hình cho MambaBlock từ mamba_minimal
    args = ModelArgs(
        d_model=d_model,
        d_state=ssm_cfg.get('d_state', 16),
        expand=ssm_cfg.get('expand', 2),
        dt_rank=ssm_cfg.get('dt_rank', 'auto'),
        d_conv=ssm_cfg.get('d_conv', 4),
        n_layer=1, vocab_size=1 
    )
    
    mixer_cls = lambda d: MambaBlock(args)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
    
    block = Block(
        d_model, mixer_cls, norm_cls=norm_cls, drop_path=drop_path,
        fused_add_norm=fused_add_norm, residual_in_fp32=residual_in_fp32,
        apply_temporal_attention=apply_temporal_attention, num_heads=num_heads, max_len=max_len
    )
    block.layer_idx = layer_idx
    return block