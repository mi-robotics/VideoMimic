from typing import Final, Optional, Type

import torch
from torch import nn as nn
from torch.nn import functional as F
import math
from einops.layers.torch import Rearrange
import logging
logger = logging.getLogger(__name__)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class Mlp(nn.Module):
    """Simple MLP with configurable hidden size and activation."""

    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None,
                 act_layer=nn.GELU, drop: float = 0.0) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class QKVAttention(nn.Module):
    """Standard Multi-head Self Attention module with separate Q, K, V projections.

    This module implements the standard multi-head attention mechanism used in transformers,
    but with separate nn.Linear layers for Q, K, and V projections ("nodified").
    It supports both the fused attention implementation (scaled_dot_product_attention) for
    efficiency when available, and a manual implementation otherwise. The module includes
    options for QK normalization, attention dropout, and projection dropout.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            scale_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Optional[Type[nn.Module]] = None,
    ) -> None:
        """Initialize the Attention module with separate Q, K, V projections.

        Args:
            dim: Input dimension of the token embeddings
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in the query, key, value projections
            qk_norm: Whether to apply normalization to query and key vectors
            proj_bias: Whether to use bias in the output projection
            attn_drop: Dropout rate applied to the attention weights
            proj_drop: Dropout rate applied after the output projection
            norm_layer: Normalization layer constructor for QK normalization if enabled
        """
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        if qk_norm or scale_norm:
            assert norm_layer is not None, 'norm_layer must be provided if qk_norm or scale_norm is True'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Separate Q, K, V projections
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(dim) if scale_norm else nn.Identity()
        self.out_proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape

        # Separate Q, K, V projections
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N, head_dim)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)


        # Check for NaNs before attention
        if torch.isnan(q).any() or torch.isnan(k).any() or torch.isnan(v).any():
            raise RuntimeError("NaN detected in q, k, or v before attention")

        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        # Check for NaNs after attention
        if torch.isnan(x).any():
            raise RuntimeError("NaN detected in output of attention")


        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.out_proj(x)
        x = self.proj_drop(x)
        return x

class QKVCrossAttention(nn.Module):
    """Multi-head cross-attention with separate Q (from tgt), K/V (from memory) projections.

    Mirrors QKVAttention but takes distinct source tensors for queries and keys/values.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            scale_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Optional[Type[nn.Module]] = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        if qk_norm or scale_norm:
            assert norm_layer is not None, 'norm_layer must be provided if qk_norm or scale_norm is True'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(dim) if scale_norm else nn.Identity()
        self.out_proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
            self,
            x: torch.Tensor,              # (B, T_tgt, C)
            memory: torch.Tensor,         # (B, T_src, C)
            attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, Tt, C = x.shape
        Ts = memory.shape[1]

        q = self.q_proj(x).reshape(B, Tt, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory).reshape(B, Ts, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory).reshape(B, Ts, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.
        )

        out = out.transpose(1, 2).reshape(B, Tt, C)
        out = self.norm(out)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out

class Block(nn.Module):
    """
    Simple transformer block
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, layer_id=0,**block_kwargs):
        super().__init__()
        self.layer_id = layer_id
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = QKVAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)


    def forward(self, x, c, **kwargs):
        if 'encoder_scale_shifts' in kwargs:
            a_shift, a_scale, ff_shift, ff_scale = kwargs['encoder_scale_shifts'][:, self.layer_id].chunk(4, dim=1)
            # x = x * scale_shifts[:, 0] + scale_shifts[:, 1]
            x = x + self.attn(modulate(self.norm1(x), a_shift, a_scale))
            x = x + self.mlp(modulate(self.norm2(x), ff_shift, ff_scale))
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x

class DecoderBlock(nn.Module):
    """Transformer decoder block with self-attention, cross-attention, and MLP."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, layer_id=0, **block_kwargs):
        super().__init__()
        self.layer_id = layer_id
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.self_attn = QKVAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)

        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = QKVCrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(
            self,
            x: torch.Tensor,
            memory: torch.Tensor,
            self_attn_mask: Optional[torch.Tensor] = None,
            cross_attn_mask: Optional[torch.Tensor] = None,
            **kwargs
    ) -> torch.Tensor:
        if 'decoder_scale_shifts' in kwargs:
            a_shift, a_scale, c_shift, c_scale, ff_shift, ff_scale = kwargs['decoder_scale_shifts'][:, self.layer_id].chunk(6, dim=1)
            # x = x * scale_shifts[:, 0] + scale_shifts[:, 1]
            x = x + self.self_attn(modulate(self.norm1(x), a_shift, a_scale), attn_mask=self_attn_mask)
            x = x + self.cross_attn(modulate(self.norm_cross(x), c_shift, c_scale), memory=memory, attn_mask=cross_attn_mask)
            x = x + self.mlp(modulate(self.norm2(x), ff_shift, ff_scale))
        else:
            x = x + self.self_attn(self.norm1(x), attn_mask=self_attn_mask)

            if memory is not None:
                x = x + self.cross_attn(self.norm_cross(x), memory=memory, attn_mask=cross_attn_mask)
                
            x = x + self.mlp(self.norm2(x))
        return x
        


class TransformerEncoder(nn.Module):
    """
    Transformer Simple transformer encoder backbone.
    """
    def __init__(
        self,
        hidden_size=512,
        num_heads=16,
        mlp_ratio=4.0,
        layers=4,
    ):
        super().__init__()
        self.num_heads = num_heads

        self.blocks = nn.ModuleList([
            Block(hidden_size, num_heads, mlp_ratio=mlp_ratio, layer_id=i) for i in range(layers)
        ])
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, None, **kwargs)
        return x

class TransformerDecoder(nn.Module):
    """Simple transformer decoder backbone with optional causal self-attention."""

    def __init__(
        self,
        hidden_size=512,
        num_heads=16,
        mlp_ratio=4.0,
        layers=4,
        causal: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.causal = causal
     
        self.blocks = nn.ModuleList([
            DecoderBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, layer_id=i) for i in range(layers)
        ])

    def _build_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        # Boolean mask where True denotes masked (disallowed) positions
        return torch.ones(T, T, dtype=torch.bool, device=device).triu(1)

    def forward(
        self,
        x: torch.Tensor,                  # (B, T_tgt, C)
        memory: torch.Tensor,             # (B, T_src, C)
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        if self.causal and tgt_mask is None:
            tgt_mask = self._build_causal_mask(x.shape[1], x.device)

        for blk in self.blocks:
            x = blk(x, memory, self_attn_mask=tgt_mask, cross_attn_mask=memory_mask, **kwargs)
        return x
    






class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb



class QKVTransformerForDiffusion(nn.Module):
    def __init__(
        self,
        input_dim, output_dim, obs_dim, emb_dim, T_obs, T_action,
        n_encoder_layers=4, n_decoder_layers=4, n_head=4,
        p_drop_emb=0.1, p_drop_attn=0.1,
        obs_type=None, causal_attn=False, past_action_visible=False
    ):
        super().__init__()
        assert T_obs is not None
        assert obs_type == 'ref', f'Only support ref type observation for now'
        self.causal_attn = causal_attn  
        self.past_action_visible = past_action_visible
      
        self.obs_dim = obs_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.T_obs = T_obs
        self.T_action = T_action
        self.obs_type = obs_type

        # Conditional encoder
        T_cond = 1 + self.T_obs
        self.time_emb = SinusoidalPosEmb(self.emb_dim)
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, self.emb_dim))
        self.cond_obs_emb = nn.Linear(self.obs_dim, self.emb_dim)
    
        self.encoder = TransformerEncoder(
            hidden_size=self.emb_dim,
            num_heads=n_head,
            mlp_ratio=4,
            layers=n_encoder_layers
        )

        # Decoder for action denoising
        self.pos_emb = nn.Parameter(torch.zeros(1, self.T_action, self.emb_dim))
        self.input_emb = nn.Linear(self.input_dim, self.emb_dim)
        self.drop = nn.Dropout(p_drop_emb)


        self.decoder = TransformerDecoder(
            hidden_size=self.emb_dim,
            num_heads=n_head,
            mlp_ratio=4,
            layers=n_decoder_layers,
            causal=False
        )

        self.ln_f = nn.LayerNorm(self.emb_dim)
        self.head = nn.Linear(self.emb_dim, output_dim)

        # Attention mask
        if self.causal_attn:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf and others (including diag) should be 0.
            mask = (torch.triu(torch.ones(self.T_action, self.T_action)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask)
        else:
            self.mask = None

        # init
        self.apply(self._init_weights)
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )



    def _init_weights(self, module):

        ignore_types = (
            nn.Dropout, 
            SinusoidalPosEmb,
            nn.TransformerEncoderLayer, 
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.MultiheadAttention,
            Rearrange,
            nn.SiLU,
            nn.Sequential,
            nn.Identity,           # add
            nn.GELU,               # add
            Block,                 # add
            DecoderBlock,          # add
            TransformerEncoder,    # add (custom)
            TransformerDecoder,    # add (custom)
            QKVAttention,          # add
            QKVCrossAttention,     # add
            Mlp,                   # add
        )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, Mlp):
            torch.nn.init.normal_(module.fc1.weight, mean=0.0, std=0.02)
            if module.fc1.bias is not None:
                torch.nn.init.zeros_(module.fc1.bias)
            torch.nn.init.normal_(module.fc2.weight, mean=0.0, std=0.02)
            if module.fc2.bias is not None:
                torch.nn.init.zeros_(module.fc2.bias)
        elif isinstance(module, QKVAttention) or isinstance(module, QKVCrossAttention):
            torch.nn.init.normal_(module.q_proj.weight, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.k_proj.weight, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.v_proj.weight, mean=0.0, std=0.02)
            if module.q_proj.bias is not None:
                torch.nn.init.zeros_(module.q_proj.bias)
            if module.k_proj.bias is not None:
                torch.nn.init.zeros_(module.k_proj.bias)
            if module.v_proj.bias is not None:
                torch.nn.init.zeros_(module.v_proj.bias)
            if module.out_proj.bias is not None:
                torch.nn.init.zeros_(module.out_proj.bias)
        elif isinstance(module, nn.MultiheadAttention):
            
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            if module.weight is not None:
                torch.nn.init.ones_(module.weight)
        elif isinstance(module, QKVTransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        elif isinstance(module, nn.Identity) or isinstance(module, nn.GELU):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def get_optim_groups(self, weight_decay):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention, QKVAttention, QKVCrossAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name  
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        no_decay.add("cond_pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def forward(self, sample, timestep, cond=None, **kwargs):
        """
        sample: (B, T_action, input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B, T_obs, cond_dim)
        """
        assert torch.is_tensor(timestep)
        assert cond.shape == (sample.shape[0], self.T_obs, self.obs_dim)
        assert sample.shape == (cond.shape[0], self.T_action, self.input_dim)
        if len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)

 
        # Encoder for conditioning
        timesteps = timestep.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        # (B, 1, obs_dim)

        cond_emb = self.cond_obs_emb(cond)

        cond_emb = torch.cat([time_emb, cond_emb], dim=1)
        # (B, T_cond, obs_dim)

        tc = cond_emb.shape[1]
        cond_pos_emb = self.cond_pos_emb[:, :tc, :]
        x = self.drop(cond_emb + cond_pos_emb)
        x = self.encoder(x, **kwargs)
        memory = x 
        # (B, T_cond, obs_dim)

        # Decoder for action prediction
        input_emb = self.input_emb(sample)

        t = sample.shape[1]
        pos_emb = self.pos_emb[:, :t, :]
        x = self.drop(input_emb + pos_emb)
        x = self.decoder(
            x=x,
            memory=memory,
            tgt_mask=self.mask,
            memory_mask=None,
            **kwargs
        )
        # (B, T, obs_dim)
        # NOTE: We don't need a memory mask because the conditioning is always on past information
        
        x = self.ln_f(x)
        x = self.head(x)
        # (B, T, output_dim)
        return x




class QKVDiffusionCloc(nn.Module):
    def __init__(
        self,
        input_dim, output_dim, obs_dim, emb_dim, T_obs, T_action,
        n_encoder_layers=4, n_decoder_layers=4, n_head=4,
        p_drop_emb=0.1, p_drop_attn=0.1,
        obs_type=None, causal_attn=False
    ):
        super().__init__()
        assert T_obs is not None
        assert obs_type == 'ref', f'Only support ref type observation for now'
        self.causal_attn = causal_attn  

      
        self.obs_dim = obs_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.emb_dim = emb_dim
        self.T_obs = T_obs
        self.T_action = T_action
        self.obs_type = obs_type


        self.time_emb = SinusoidalPosEmb(self.emb_dim)
        # Decoder for action denoising
        self.pos_emb = nn.Parameter(torch.zeros(1, self.T_action*2, self.emb_dim))
        self.action_emb = nn.Linear(self.input_dim, self.emb_dim)
        self.obs_emb = nn.Linear(self.obs_dim, self.emb_dim)
        self.drop = nn.Dropout(p_drop_emb)


        self.decoder = TransformerDecoder(
            hidden_size=self.emb_dim,
            num_heads=n_head,
            mlp_ratio=4, #TODO should change this, manually set ff sizes, limits the embed dim
            layers=n_decoder_layers,
            causal=False
        )

        self.ln_action = nn.LayerNorm(self.emb_dim)
        self.ln_obs = nn.LayerNorm(self.emb_dim)
        self.head_action = nn.Linear(self.emb_dim, output_dim)
        self.head_obs = nn.Linear(self.emb_dim, self.obs_dim)

        # Attention mask
        if self.causal_attn:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf and others (including diag) should be 0.
            mask = (torch.triu(torch.ones(self.T_action, self.T_action)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask)
        else:
            self.mask = None

        # init
        self.apply(self._init_weights)
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )



    def _init_weights(self, module):

        ignore_types = (
            nn.Dropout, 
            SinusoidalPosEmb,
            nn.TransformerEncoderLayer, 
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.MultiheadAttention,
            Rearrange,
            nn.SiLU,
            nn.Sequential,
            nn.Identity,           # add
            nn.GELU,               # add
            Block,                 # add
            DecoderBlock,          # add
            TransformerEncoder,    # add (custom)
            TransformerDecoder,    # add (custom)
            QKVAttention,          # add
            QKVCrossAttention,     # add
            Mlp,                   # add
        )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, Mlp):
            torch.nn.init.normal_(module.fc1.weight, mean=0.0, std=0.02)
            if module.fc1.bias is not None:
                torch.nn.init.zeros_(module.fc1.bias)
            torch.nn.init.normal_(module.fc2.weight, mean=0.0, std=0.02)
            if module.fc2.bias is not None:
                torch.nn.init.zeros_(module.fc2.bias)
        elif isinstance(module, QKVAttention) or isinstance(module, QKVCrossAttention):
            torch.nn.init.normal_(module.q_proj.weight, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.k_proj.weight, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.v_proj.weight, mean=0.0, std=0.02)
            if module.q_proj.bias is not None:
                torch.nn.init.zeros_(module.q_proj.bias)
            if module.k_proj.bias is not None:
                torch.nn.init.zeros_(module.k_proj.bias)
            if module.v_proj.bias is not None:
                torch.nn.init.zeros_(module.v_proj.bias)
            if module.out_proj.bias is not None:
                torch.nn.init.zeros_(module.out_proj.bias)
        elif isinstance(module, nn.MultiheadAttention):
            
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            if module.weight is not None:
                torch.nn.init.ones_(module.weight)
        elif isinstance(module, QKVTransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        elif isinstance(module, nn.Identity) or isinstance(module, nn.GELU):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def get_optim_groups(self, weight_decay):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention, QKVAttention, QKVCrossAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name  
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        no_decay.add("cond_pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def forward(self, sample, timestep, cond=None, **kwargs):
        """
        sample: (B, T_action, input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B, T_obs, cond_dim)
        """
        assert torch.is_tensor(timestep)
        assert cond.shape == (sample.shape[0], self.T_obs, self.obs_dim+self.input_dim)
        assert sample.shape == (cond.shape[0], self.T_action, self.input_dim+self.obs_dim)
        if len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)

 
        # Encoder for conditioning
        timesteps = timestep.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        # (B, 1, obs_dim)


        full_sample = torch.cat((cond, sample), dim=1)
        actions = full_sample[:, :, :self.input_dim]
        obs = full_sample[:, :, self.input_dim:]

        # Decoder for action prediction
        action_emb = self.action_emb(actions) # [bs, seq, emb_dim]
        obs_emb = self.obs_emb(obs) # [bs, seq, emb_dim]

        sa_seq = torch.cat((obs_emb.unqueeze(-2), action_emb.unqueeze(-2)), dim=-2).flatten(1,2) # [bs, seq*2, emb_dim] --> s, a, s, a 

        t = sa_seq.shape[1]
        pos_emb = self.pos_emb[:, :t, :]
        x = self.drop(sa_seq + pos_emb)

        x = self.decoder(
            x=x,
            memory=None,
            tgt_mask=self.mask,
            memory_mask=None,
            **kwargs
        )
        
        x_obs = x[:, 0::2, :]
        x_obs = x_obs[:, :self.T_obs, :]

        x_action = x[:, 1::2, :]
        x_action = x_action[:, :self.T_obs, :]


        x_action = self.head_action(self.ln_action(x_action))
        x_obs = self.head_obs(self.ln_obs(x_obs))

        x_out = torch.cat((x_action, x_obs), dim=-1)
        # (B, T, output_dim)
        return x_out






class QKVMetaDiffusion(nn.Module):
    def __init__(
        self,
        input_dim, output_dim, obs_dim, emb_dim, T_obs, T_action,
        n_encoder_layers=4, n_decoder_layers=4, n_head=4,
        p_drop_emb=0.1, p_drop_attn=0.1,
        obs_type=None, causal_attn=False, past_action_visible=False, causal_attn_type=None,
        learn_latent=False,
        is_variational=False, kl_beta=1, latent_size=64,
        normalize_latent=False, mmd_weight=0.0, is_enc_past_visible=False,
        condition_mechanism=None
    ):
        super().__init__()
        '''
        sample: (B, T_action, input_dim)

        '''
        assert T_obs is not None
        assert obs_type == 'ref', f'Only support ref type observation for now'
        self.causal_attn = causal_attn  
        self.causal_attn_type = causal_attn_type
        self.past_action_visible = past_action_visible
        self.obs_dim = obs_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.T_obs = T_obs
        self.T_action = T_action
        self.obs_type = obs_type
        self.is_variational = is_variational
        self.kl_beta = kl_beta
        self.learn_latent = learn_latent
        self.latent_size = latent_size
        self.normalize_latent = normalize_latent
        self.mmd_weight = mmd_weight
        self.is_enc_past_visible = is_enc_past_visible
        self.condition_mechanism = condition_mechanism
        # Conditional encoder
        self.time_emb = SinusoidalPosEmb(self.emb_dim)
        self.pos_emb_dec = nn.Parameter(torch.zeros(1,(  self.T_obs*2+self.T_action*2)+1, self.emb_dim))
        self.action_emb_dec = nn.Linear(self.input_dim, self.emb_dim)
        self.obs_emb_dec = nn.Linear(self.obs_dim, self.emb_dim)

        if self.learn_latent:
            self.action_emb_enc= nn.Linear(self.input_dim, self.emb_dim)
            self.obs_emb_enc = nn.Linear(self.obs_dim, self.emb_dim)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.emb_dim))
            self.pos_emb_enc = nn.Parameter(torch.zeros(1,( self.T_obs*2+self.T_action*2)+1, self.emb_dim))
        
            if self.is_variational:
                self.cls_token_proj = nn.Linear(self.emb_dim, self.latent_size*2)
            else:
                self.cls_token_proj = nn.Linear(self.emb_dim, self.latent_size)

        
            self.encoder = TransformerEncoder(
                hidden_size=self.emb_dim,
                num_heads=n_head,
                mlp_ratio=4,
                layers=n_encoder_layers
            )

        # Decoder for action denoising
        self.input_emb = nn.Linear(self.input_dim, self.emb_dim)
        self.drop = nn.Dropout(p_drop_emb)
        self.latent_up_proj = nn.Linear(self.latent_size, self.emb_dim)
        self.decoder = TransformerDecoder(
            hidden_size=self.emb_dim,
            num_heads=n_head,
            mlp_ratio=4,
            layers=n_decoder_layers
        )

        self.ln_action = nn.LayerNorm(self.emb_dim)
        self.ln_obs = nn.LayerNorm(self.emb_dim)
        self.head_action = nn.Linear(self.emb_dim, output_dim)
        self.head_obs = nn.Linear(self.emb_dim, self.obs_dim)

        # Attention mask
        if self.causal_attn:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf and others (including diag) should be 0.
        
            decoder_size =  self.T_obs*2 + self.T_action*2
            if self.condition_mechanism == 'cat' and self.learn_latent:
                decoder_size += 1

            if self.causal_attn_type == 'cloc':
                pass
            elif self.causal_attn_type == 'default':
                mask = (torch.triu(torch.ones(decoder_size, decoder_size)) == 1).transpose(0, 1)
                mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
                self.register_buffer("mask", mask)
            else:
                raise ValueError(f"Unsupported causal attention type: {self.causal_attn_type}")
        else:
            self.mask = None

        # init
        self.apply(self._init_weights)
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )



    def _init_weights(self, module):

        ignore_types = (
            nn.Dropout, 
            SinusoidalPosEmb,
            nn.TransformerEncoderLayer, 
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.MultiheadAttention,
            Rearrange,
            nn.SiLU,
            nn.Sequential,
            nn.Identity,           # add
            nn.GELU,               # add
            Block,                 # add
            DecoderBlock,          # add
            TransformerEncoder,    # add (custom)
            TransformerDecoder,    # add (custom)
            QKVAttention,          # add
            QKVCrossAttention,     # add
            Mlp,                   # add
        )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, Mlp):
            torch.nn.init.normal_(module.fc1.weight, mean=0.0, std=0.02)
            if module.fc1.bias is not None:
                torch.nn.init.zeros_(module.fc1.bias)
            torch.nn.init.normal_(module.fc2.weight, mean=0.0, std=0.02)
            if module.fc2.bias is not None:
                torch.nn.init.zeros_(module.fc2.bias)
        elif isinstance(module, QKVAttention) or isinstance(module, QKVCrossAttention):
            torch.nn.init.normal_(module.q_proj.weight, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.k_proj.weight, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.v_proj.weight, mean=0.0, std=0.02)
            if module.q_proj.bias is not None:
                torch.nn.init.zeros_(module.q_proj.bias)
            if module.k_proj.bias is not None:
                torch.nn.init.zeros_(module.k_proj.bias)
            if module.v_proj.bias is not None:
                torch.nn.init.zeros_(module.v_proj.bias)
            if module.out_proj.bias is not None:
                torch.nn.init.zeros_(module.out_proj.bias)
        elif isinstance(module, nn.MultiheadAttention):
            
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            if module.weight is not None:
                torch.nn.init.ones_(module.weight)
        elif isinstance(module, QKVTransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, QKVMetaDiffusion):
            if self.learn_latent:
                torch.nn.init.normal_(module.pos_emb_enc, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.pos_emb_dec, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        elif isinstance(module, nn.Identity) or isinstance(module, nn.GELU):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def get_optim_groups(self, weight_decay):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention, QKVAttention, QKVCrossAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name  
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb_dec")
        # no_decay.add("cond_pos_emb")
        if self.learn_latent:
            no_decay.add("cls_token")
            no_decay.add("pos_emb_enc")
        
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def reparameterize(self, mean, logvar):
        return mean + torch.randn_like(mean) * torch.exp(logvar * 0.5)
    
    def forward(self, sample_clean, sample, timestep, cond=None, **kwargs):
        """
        sample: (B, T_action, input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B, T_obs, cond_dim)
        """
        assert torch.is_tensor(timestep)
        assert cond.shape[1:] == (self.T_obs, self.input_dim+self.obs_dim)
        assert sample.shape[1:] == (self.T_action+1, self.input_dim+self.obs_dim)
        if len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)

 
        # Encoder for conditioning
        timesteps = timestep.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        # (B, 1, obs_dim)

        if self.learn_latent:   
            actions = sample_clean[:, 1:, :self.input_dim]
            obs = sample_clean[:, 1:, self.input_dim:]
            actions_cond = cond[:, :, :self.input_dim]
            obs_cond = cond[:, :, self.input_dim:]

            if self.is_enc_past_visible:
                actions = torch.cat((actions_cond, actions), dim=1)
                obs = torch.cat((obs_cond, obs), dim=1)
                # Decoder for action prediction
                action_emb = self.action_emb_enc(actions) # [bs, seq, emb_dim]
                obs_emb = self.obs_emb_enc(obs) # [bs, seq, emb_dim]

                assert actions.shape[1] == self.T_action + self.T_obs
                assert obs.shape[1] == self.T_action + self.T_obs
            else:
                actions = sample_clean[:, :, :self.input_dim]
                obs = sample_clean[:, :, self.input_dim:]
                action_emb = self.action_emb_enc(actions) # [bs, seq, emb_dim]
                obs_emb = self.obs_emb_enc(obs) # [bs, seq, emb_dim]

                assert actions.shape[1] == self.T_action+1 
                assert obs.shape[1] == self.T_action + 1


            x_in = torch.cat((obs_emb.unsqueeze(-2), action_emb.unsqueeze(-2)), dim=-2).flatten(1,2) # [bs, seq*2, emb_dim] --> s, a, s, a 

            x_in = x_in 
            x_in = torch.cat((
                self.cls_token.repeat(x_in.shape[0], 1, 1),
                x_in,
            ), dim=1)
            # (B, T_cond, obs_dim)

            t = x_in.shape[1]
            pos_emb = self.pos_emb_enc[:, :t, :]
            x_in = self.drop(x_in + pos_emb)
            x_in = self.encoder(x_in, **kwargs)

            cls_emb = x_in[:, 0:1, :]

            if self.is_variational:
                cls_emb = self.cls_token_proj(cls_emb)
                cls_mean, cls_logvar = cls_emb.chunk(2, dim=-1)
                cls_emb = self.reparameterize(cls_mean, cls_logvar)
            else:
                cls_emb = self.cls_token_proj(cls_emb)
            ############# Denoiser #############

            if self.normalize_latent:
                cls_emb = cls_emb / cls_emb.norm(dim=-1, keepdim=True)


        actions = sample[:, :, :self.input_dim]
        obs = sample[:, 1:, self.input_dim:]
        actions_cond = cond[:, :-1, :self.input_dim]
        obs_cond = cond[:, :, self.input_dim:]

        actions = torch.cat((actions_cond, actions), dim=1)
        obs = torch.cat((obs_cond, obs), dim=1)

        assert actions.shape[1] == self.T_action + self.T_obs
        assert obs.shape[1] == self.T_action + self.T_obs
        # Decoder for action prediction
        action_emb = self.action_emb_dec(actions) # [bs, seq, emb_dim]
        obs_emb = self.obs_emb_dec(obs) # [bs, seq, emb_dim]

        x_in = torch.cat((obs_emb.unsqueeze(-2), action_emb.unsqueeze(-2)), dim=-2).flatten(1,2) # [bs, seq*2, emb_dim] --> s, a, s, a 
        x_in = x_in + time_emb

        memory = None
        if self.learn_latent:
            if self.condition_mechanism == 'cat':
                x_in = torch.cat((
                    self.latent_up_proj(cls_emb),
                    x_in,
                ), dim=1)
            elif self.condition_mechanism == 'cross_attn':
                memory = self.latent_up_proj(cls_emb)
            else:
                raise ValueError(f"Unsupported condition mechanism: {self.condition_mechanism}")

        t = x_in.shape[1]
        pos_emb = self.pos_emb_dec[:, :t, :]
        x_in = self.drop(x_in + pos_emb)

        if torch.isnan(x_in).any():
            raise ValueError("NaNs detected in x_in tensor before decoder.")

        x = self.decoder(
            x=x_in,
            tgt_mask=self.mask,
            memory_mask=None,
            memory=memory,
            **kwargs
        )

        if torch.isnan(x).any():
            print('mask----------',self.mask)
            input()
            raise ValueError("NaNs detected in x tensor after decoder.")

        if self.condition_mechanism == 'cat':
            x = x[:, 1:, :]
     
        # (B, T, obs_dim)
        x_obs = x[:, 0::2, :]
        x_obs = x_obs[:, self.T_obs-1:, :]
        x_action = x[:, 1::2, :]
        x_action = x_action[:, self.T_obs-1:, :]


        x_action = self.head_action(self.ln_action(x_action))
        x_obs = self.head_obs(self.ln_obs(x_obs))

        x_out = torch.cat((x_action, x_obs), dim=-1)
        return x_out, {
            'cls_mean': cls_mean if self.is_variational and self.learn_latent else None,
            'cls_logvar': cls_logvar if self.is_variational and self.learn_latent else None,
            'cls_emb': cls_emb if self.learn_latent else None
        }