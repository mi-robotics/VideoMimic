from typing import Final, Optional, Type

import torch
from torch import nn as nn
from torch.nn import functional as F
import math
from einops.layers.torch import Rearrange
import logging
logger = logging.getLogger(__name__)


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


        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )


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
            dropout_p=self.attn_drop.p if self.training else 0.,
            is_causal=True
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
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = QKVAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)


    def forward(self, x, c):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class DecoderBlock(nn.Module):
    """Transformer decoder block with self-attention, cross-attention, and MLP."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
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
    ) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), attn_mask=self_attn_mask)
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
            Block(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, None)
        return x

class TransformerDecoder(nn.Module):
    """Simple transformer decoder backbone with optional causal self-attention."""

    def __init__(
        self,
        hidden_size=512,
        num_heads=16,
        mlp_ratio=4.0,
        layers=4,
        causal: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.causal = causal
        self.blocks = nn.ModuleList([
            DecoderBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(layers)
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
    ) -> torch.Tensor:
        if self.causal and tgt_mask is None:
            tgt_mask = self._build_causal_mask(x.shape[1], x.device)

        for blk in self.blocks:
            x = blk(x, memory, self_attn_mask=tgt_mask, cross_attn_mask=memory_mask)
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
        obs_type=None, causal_attn=False, past_action_visible=False,
        ####
        use_image_conds=False, image_cond_method=None, image_encoder=None,
        image_embedding_units=None,
        finetune_node=None
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

        ####
        ####
        ####
        ####
        ####
        self.use_image_conds = use_image_conds
        self.image_encoder = image_encoder
        self.image_cond_method = image_cond_method
        self.image_embedding_units = image_embedding_units
        self.finetune_node = finetune_node
        if self.use_image_conds:
            assert self.finetune_node in ['encoder', 'encoder_mdm']
            assert self.image_cond_method in ['pdp_encoder', 'pdp_decoder', 'film_encoder', 'film_decoder', 'adaln_encoder', 'adaln_decoder']   
            assert self.image_encoder in ['dinov2', 'dinov3']
            assert len(self.image_embedding_units) >= 1

            self.image_backbone_fn, img_enc_conf = None, None, None # load_image_encoder(self.image_encoder)
            img_encoder_layers = []
            in_units = 2*img_enc_conf['emb_dim']
            for unit in self.image_embedding_units:
                img_encoder_layers.append(torch.nn.Linear(in_units, unit))
                img_encoder_layers.append(torch.nn.GeLU())
                img_encoder_layers.append(torch.nn.Dropout(0.1))
                in_units = unit
            img_encoder_layers.append(torch.nn.Linear(in_units, self.emb_dim))
            self.image_encoder = torch.nn.Sequential(*img_encoder_layers)

            if 'pdp' in self.image_cond_method:
                self.film = torch.nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(self.emb_dim, 2*self.emb_dim, bias=True)
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

        if self.use_image_conds:
            assert 'image_conds' in kwargs
            #TODO two image must come in
            img = kwargs['image_conds'].to(sample.device)
            img_emb = self.image_backbone_fn(img)
            img_emb_0 = img_emb[0::2]
            img_emb_1 = img_emb[1::2]
            img_emb = torch.cat([img_emb_0, img_emb_1], dim=-1)
            img_emb = self.image_encoder(img_emb)


        # Encoder for conditioning
        timesteps = timestep.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        # (B, 1, obs_dim)

        cond_emb = self.cond_obs_emb(cond)

        if self.use_image_conds and self.image_cond_method == 'pdp_encoder':
            scale, shift = self.film(img_emb).chunk(2, dim=-1)
            cond_emb = cond_emb * (scale.unsqueeze(0) + 1) + shift.unsqueeze(0)
         

        cond_emb = torch.cat([time_emb, cond_emb], dim=1)
        # (B, T_cond, obs_dim)

        tc = cond_emb.shape[1]
        cond_pos_emb = self.cond_pos_emb[:, :tc, :]
        x = self.drop(cond_emb + cond_pos_emb)
        x = self.encoder(x)
        memory = x 
        # (B, T_cond, obs_dim)

        # Decoder for action prediction
        input_emb = self.input_emb(sample)
        if self.use_image_conds and self.image_cond_method == 'pdp_decoder':
            scale, shift = self.film(img_emb).chunk(2, dim=-1)
            input_emb = input_emb * (scale.unsqueeze(0) + 1) + shift.unsqueeze(0)

        t = sample.shape[1]
        pos_emb = self.pos_emb[:, :t, :]
        x = self.drop(input_emb + pos_emb)
        x = self.decoder(
            x=x,
            memory=memory,
            tgt_mask=self.mask,
            memory_mask=None
        )
        # (B, T, obs_dim)
        # NOTE: We don't need a memory mask because the conditioning is always on past information
        
        x = self.ln_f(x)
        x = self.head(x)
        # (B, T, output_dim)
        return x
