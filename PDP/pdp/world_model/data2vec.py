import torch
import torch.nn as nn

import numpy as np
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
import math

from pdp.utils.normalizer import LinearNormalizer


from dataclasses import dataclass, field
from typing import Optional, Any, Dict

from pdp.modules_v2 import TransformerEncoder, ResidualMLP
from pdp.world_model.d2vEma import EMA
from typing import Dict, Iterable, List, Optional, Tuple

@dataclass
class SequenceEncoderConfig:
    state_dim: int
    action_dim: int
    hidden_size: int


@dataclass
class Data2VecConfig:
    sequence_encoder_config: SequenceEncoderConfig
    num_layers: int
    num_heads: int
    mlp_ratio: int
    hidden_size: int
    regressor_size: str

    mask_prob_start: float #0.48
    mask_prob_end: float #0.48
    mask_prob_anneal_end_step: int
    mask_noise_std: float #0.01

    loss_beta: float
    loss_scale: float

    ema_decay: float
    ema_end_decay: float
    ema_anneal_end_step: int

    average_top_k_layers: int
    layer_norm_target_layer: bool
    instance_norm_target_layer: bool
    batch_norm_target_layer: bool
    layer_norm_targets: bool
    instance_norm_targets: bool





class SequenceEncoder(nn.Module):
    def __init__(self, 
        config: SequenceEncoderConfig
        ):
        super().__init__()
        self.state_encoder = ResidualMLP(config.state_dim, config.hidden_size)
        self.action_encoder = ResidualMLP(config.action_dim, config.hidden_size)
    
    def forward(self, state, action):
        '''
        state: (B, T_state, state_dim)
        action: (B, T_action, action_dim)
        '''
        state = self.state_encoder(state)
        action = self.action_encoder(action)
        return state, action


class Data2Vec(nn.Module):
    def __init__(self, 
        config: Data2VecConfig
        ):

        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.sequence_encoder = SequenceEncoder(config.sequence_encoder_config)
        self.transformer_encoder = TransformerEncoder(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            layers=config.num_layers,
            **{
                'proj_drop': config.proj_drop,
                'attn_drop': config.attn_drop,
            }
        )
        self._build_regressor(config.regressor_size)

        self.learned_mask = nn.Parameter(torch.zeros(config.hidden_size))

        self.mask_prob_start = config.mask_prob_start
        self.mask_prob_end = config.mask_prob_end
        self.mask_prob_anneal_end_step = config.mask_prob_anneal_end_step
        self.mask_prob_updates = 0
        self.mask_noise_std = config.mask_noise_std

        self.ema = EMA(self.transformer_encoder, config)  # EMA acts as the teacher
        self.ema_decay = config.ema_decay
        self.ema_end_decay = config.ema_end_decay
        self.ema_anneal_end_step = config.ema_anneal_end_step

        return


    def _build_regressor(self, size: str):
        assert size in ['sm', 'md', 'bg']

        if size == 'sm':
            self.regressor = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Dropout(self.config.representation_drop)
            )
        elif size == 'md':
            self.regressor = nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size * 2),
                    nn.GELU(),
                    nn.Linear(self.hidden_size * 2, self.hidden_size),
                    nn.Dropout(self.config.representation_drop)
            )
        elif size == 'bg':
            self.regressor = nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size * 2),
                    nn.GELU(),
                    nn.Linear(self.hidden_size * 2, self.hidden_size*2),
                    nn.GELU(),
                    nn.Linear(self.hidden_size * 2, self.hidden_size),
                    nn.Dropout(self.config.representation_drop)
            )

    def get_optim_groups(self, weight_decay):
 
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear)
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

                elif 'rnn' in fpn:
                    no_decay.add(fpn)


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

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer = normalizer


    def ema_step(self):
        """
        One EMA step for the offline model until the ending decay value is reached
        """
        if self.ema_decay != self.ema_end_decay:
            if self.ema.num_updates >= self.ema_anneal_end_step:
                decay = self.ema_end_decay
            else:
                decay = self.ema.get_annealed_rate(
                    self.ema_decay,
                    self.ema_end_decay,
                    self.ema.num_updates,
                    self.ema_anneal_end_step,
                )
            self.ema.decay = decay
        if self.ema.decay < 1:
            self.ema.step(self.transformer_encoder)

    def _current_prob(self, update=True):
        if self.mask_prob_anneal_end_step <= 0:
            p = self.mask_prob_end
        else:
            frac = min(1.0, self.mask_prob_updates / self.mask_prob_anneal_end_step)
            p = self.mask_prob_start + (self.mask_prob_end - self.mask_prob_start) * frac
        if update:
            self.mask_prob_updates += 1
        return float(max(0.0, min(1.0, p)))


    def mask_state(self, states: torch.Tensor):
        """
        states: (B, T, F)
        Returns:
          masked_states: (B, T, F)
          mask_bt: (B, T) True where a timestep was masked/noised
        """
        B, T, F = states.shape
        assert F == self.learned_mask.numel(), "hidden_size must match states' last dim"

        # Bernoulli per timestep
        p = self._current_prob()
        mask_bt = (torch.rand(B, T, device=states.device) < p)   # (B, T)
        mask_btf = mask_bt.unsqueeze(-1)                          # (B, T, 1)

        # Gaussian noise per (B,T,F)
        noise = torch.randn(B, T, F, device=states.device, dtype=states.dtype) * self.mask_noise_std

        # Target value for masked steps
        target = self.learned_mask.view(1, 1, F).to(states.device, states.dtype)

        masked_value = target + noise
        masked_states = torch.where(mask_btf, masked_value, states)

        return masked_states, mask_bt


    def normalize_target_layers(self, layer_embeddings, k):

        """
        layer_embeddings: list[Tensor], each (B, T, C) from lower->higher layer
        k: average the last k layers
        cfg: has flags:
            - batch_norm_target_layer, instance_norm_target_layer, layer_norm_target_layer
            - layer_norm_targets, instance_norm_targets
        Returns:
            y: Tensor (B, T, C)
        """
        # take last k layers, each (B, T, C)
        ys = layer_embeddings[-k:]

        # If we will apply per-layer BN/IN, move to (B, C, T)
        need_bct = bool(self.config.instance_norm_target_layer or self.config.batch_norm_target_layer)
        if need_bct:
            ys = [tl.permute(0, 2, 1).contiguous() for tl in ys]  # (B, T, C) -> (B, C, T)

        # Per-layer norms (operate on each layer separately)
        if self.config.batch_norm_target_layer:
            # Functional BN with batch stats per forward: input (N, C, L)
            ys = [
                F.batch_norm(
                    tl.float(),
                    running_mean=None, running_var=None,
                    training=True, momentum=0.1, eps=1e-5
                )
                for tl in ys
            ]

        if self.config.instance_norm_target_layer:
            # IN expects (N, C, L)
            ys = [F.instance_norm(tl.float(), eps=1e-5) for tl in ys]

        # Bring back to (B, T, C) if we went to (B, C, T)
        if need_bct:
            ys = [tl.permute(0, 2, 1).contiguous() for tl in ys]  # (B, C, T) -> (B, T, C)

        # Optional per-layer LN (on last dim C)
        if self.config.layer_norm_target_layer:
            ys = [F.layer_norm(tl.float(), normalized_shape=(tl.shape[-1],), eps=1e-5) for tl in ys]

        # Average across the k layers -> (B, T, C)
        y = sum(ys) / len(ys)

        # Target-level norms (operate on averaged representation)
        if self.config.layer_norm_targets:
            y = F.layer_norm(y.float(), normalized_shape=(y.shape[-1],), eps=1e-5)  # (B, T, C)

        if self.config.instance_norm_targets:
            # IN over channels per (B, C, T): transpose, norm, transpose back
            y = F.instance_norm(y.transpose(1, 2).contiguous().float(), eps=1e-5).transpose(1, 2).contiguous()

        return y

    def criterion(self, x, y):

        sz = x.size(-1)
        if self.config.loss_beta == 0:
            loss = F.mse_loss(x.float(), y.float(), reduction="none").sum(dim=-1)
        else:
            loss = F.smooth_l1_loss(
                x.float(), y.float(), reduction="none", beta=self.config.loss_beta
            ).sum(dim=-1)

        result = {
            "losses": {
                "main": loss.sum() / math.sqrt(sz)
                if self.config.loss_scale <= 0
                else loss.sum() * self.config.loss_scale,
            },
            "sample_size": loss.numel(),
        }

        # logging other values
        other_logs = {
            "ema_decay": self.ema.get_decay() * 1000,
            "loss_mean": loss.mean()
        }
        result["logs"] = other_logs
        return result


    def forward(self, batch):
        return self.compute_loss(batch)

    def compute_loss(self, batch):
        """
        Computes the main loss for the Data2Vec model given a batch of data.

        This function performs the following steps:
        1. Normalizes the input batch (observations and actions).
        2. Encodes the states and actions using the sequence encoder.
        3. Applies masking to the state embeddings.
        4. Concatenates state and action embeddings to form the input for the transformer encoder (student).
        5. Runs the transformer encoder to obtain student representations.
        6. Uses the EMA (teacher) model to obtain target representations from the unmasked input.
        7. Normalizes and averages the top-k teacher layers as targets.
        8. Selects only the masked positions for loss computation.
        9. Passes student representations through the regressor head.
        10. Computes the loss between student and teacher representations.
        11. Returns the main loss and logging information.

        Args:
            batch (dict): A dictionary containing 'obs' and 'action' tensors.

        Returns:
            tuple: (main loss, logs dictionary)
        """


        # model forward in online mode (student)
        nbatch = self.normalizer.normalize({
            'obs': batch['obs'],
            'action': batch['action']
        })  
        
        states_emb, actions_emb = self.sequence_encoder(nbatch['obs'], nbatch['action'])
        target_states_emb = states_emb.clone()
        assert states_emb.size(1) == actions_emb.size(1), "Interleaving requires T_state == T_action"
        

        states_emb, mask_bt = self.mask_state(states_emb)

        src = torch.cat((states_emb.unsqueeze(-2), actions_emb.unsqueeze(-2)), dim=-2).flatten(1,2) # (B, T_state+T_action, hidden_size) --> s, a, s, a 
        trg = torch.cat((target_states_emb.unsqueeze(-2), actions_emb.unsqueeze(-2)), dim=-2).flatten(1,2) # (B, T_state+T_action, hidden_size) --> s, a, s, a 

        x, _ = self.transformer_encoder(src, return_extras=True) # fetch the last layer outputs

    
        # model forward in offline mode (teacher)
        with torch.no_grad():
            self.ema.model.eval()
            _, layer_embeddings = self.ema.model(trg, return_extras=True)  # fetch the last transformer layers outputs
    
            y = self.normalize_target_layers(layer_embeddings, self.config.average_top_k_layers)
  
        x = x[:, 0::2][mask_bt] #only predicting the states
        y = y[:, 0::2][mask_bt] #only predicting the states

        x = self.regressor(x)

        out = self.criterion(x, y)

        out['logs'].update(log_mask_stats(mask_bt, self._current_prob(update=False)))
        out['logs'].update(log_rep_stats('student_rep_stats', x))
        out['logs'].update(log_rep_stats('teacher_rep_stats', y))
        out['logs'].update(cosine_and_l2('student_teacher_similarity', x, y))
        out['logs'].update(ema_stats(self.ema, self))

        out['logs'].update(attn_entropy_per_layer(layer_embeddings))
        out['logs'].update(learned_mask_stats(self.learned_mask))

        return out['losses']['main'], out['logs']


    




###############################
###############################
###############################
###############################
# LOGGING UTILS
###############################
###############################
###############################
###############################



def log_mask_stats(mask_bt: torch.Tensor, scheduled_p: float) -> Dict[str, float]:
    """
    mask_bt: (B, T) boolean mask used for states
    scheduled_p: scalar p you intended to use
    """
    with torch.no_grad():
        actual = mask_bt.float().mean().item()
    return {
        "mask/scheduled_p": float(scheduled_p),
        "mask/actual_frac": actual,
        "mask/delta": float(actual - scheduled_p),
    }

def log_rep_stats(name: str, reps_btc: torch.Tensor) -> Dict[str, float]:
    """
    reps_btc: (B, T, C) representations (student or teacher)
    """
    x = reps_btc.detach()
    mean = x.mean().item()
    std = x.std(unbiased=False).item()
    # channelwise variance averaged (helps detect collapse in C)
    ch_var = x.var(dim=(0,1), unbiased=False).mean().item()
    return {
        f"{name}/mean": mean,
        f"{name}/std": std,
        f"{name}/ch_var": ch_var,
        f"{name}/norm_mean": x.norm(dim=-1).mean().item(),
    }

def select_masked_positions(reps_btc: torch.Tensor, mask_bt: torch.Tensor) -> torch.Tensor:
    """
    Returns masked positions flattened to (N_masked, C) for convenient similarity stats.
    reps_btc: (B, T, C)
    mask_bt: (B, T) boolean
    """
    B, T, C = reps_btc.shape
    assert mask_bt.shape == (B, T)
    return reps_btc[mask_bt]  # (N_masked, C)

def cosine_and_l2(name: str, x_nc: torch.Tensor, y_nc: torch.Tensor) -> Dict[str, float]:
    """
    x_nc, y_nc: (N, C) masked student vs teacher vectors, aligned by position.
    """
    assert x_nc.shape == y_nc.shape
    x = F.normalize(x_nc.detach(), dim=-1)
    y = F.normalize(y_nc.detach(), dim=-1)
    cos = (x * y).sum(dim=-1)
    l2 = (x_nc.detach() - y_nc.detach()).pow(2).sum(dim=-1).sqrt()
    return {
        f"{name}/cos_mean": cos.mean().item(),
        f"{name}/cos_med": cos.median().item(),
        f"{name}/l2_mean": l2.mean().item(),
        f"{name}/l2_med": l2.median().item(),
    }




def ema_stats(ema_obj, student_model: torch.nn.Module) -> Dict[str, float]:
    """
    ema_obj: your EMA wrapper
    student_model: the model being tracked
    """
    out = {}
    # decay and updates if available
    if hasattr(ema_obj, "get_decay"):
        out["ema/decay"] = float(ema_obj.get_decay())
    if hasattr(ema_obj, "num_updates"):
        out["ema/num_updates"] = float(getattr(ema_obj, "num_updates"))
    # parameter drift
    with torch.no_grad():
        total, cnt = 0.0, 0
        for (n_s, p_s), (n_t, p_t) in zip(student_model.named_parameters(),
                                          ema_obj.model.named_parameters()):
            if p_s.shape != p_t.shape:
                continue
            d = (p_s.detach() - p_t.detach()).pow(2).mean().sqrt().item()
            total += d
            cnt += 1
        if cnt > 0:
            out["ema/param_rms_l2"] = float(total / cnt)
    return out

def attn_entropy_per_layer(attn_weights: List[torch.Tensor]) -> Dict[str, float]:
    """
    attn_weights: list of (B, H, T, T) tensors for each layer, if your encoder returns them.
    """
    logs = {}
    for i, w in enumerate(attn_weights):
        w = w.detach()
        # small epsilon for numerical stability
        p = w.clamp_min(1e-9)
        ent = -(p * p.log()).sum(dim=-1).mean()  # average over keys, heads, batch, queries
        logs[f"attn/layer_{i}_entropy"] = ent.item()
    return logs

def learned_mask_stats(mask_vec: torch.Tensor) -> Dict[str, float]:
    """
    mask_vec: (C,) parameter 'learned_mask'
    """
    x = mask_vec.detach()
    return {
        "lmask/mean": x.mean().item(),
        "lmask/std": x.std(unbiased=False).item(),
        "lmask/norm": x.norm().item(),
    }


"""

# ✅ What “Good Learning” Looks Like

## Masking dynamics
- **Scheduled mask prob `p`**
  - _Healthy:_ Smoothly follows your anneal schedule (monotonic if designed that way).
  - _Red flags:_ Abrupt jumps; oscillations; never reaches target end value.
- **Actual masked fraction**
  - _Healthy:_ ≈ scheduled `p` (±1–2%).
  - _Red flags:_ Persistent mismatch → bug in mask generation/broadcasting.
- **Mask noise std**
  - _Healthy:_ Small and stable (e.g., 0.01).
  - _Red flags:_ Zero (no stochasticity) or too large (targets dominated by noise).

## Representation health / collapse checks
- **Student/Teacher mean & std; channel variance**
  - _Healthy:_ Non-zero std that stabilizes; channel variance not collapsing to ~0.
  - _Red flags:_ Std → 0 (collapse) or explodes; mean drifting far from 0 without normalization.
- **Cosine similarity on masked positions**
  - _Healthy:_ Starts ~0.1–0.4 and steadily rises toward ~0.6–0.9.
  - _Red flags:_ Flat near 0 (no learning) or instantly ≈1.0 (data leakage/shortcut).
- **L2 distance on masked positions**
  - _Healthy:_ Clear decreasing trend; diminishing returns later.
  - _Red flags:_ Flat or increasing; large stepwise jumps (instability).

## EMA health
- **EMA decay**
  - _Healthy:_ Follows anneal schedule; typically high (0.99–0.999+) after warm-up.
  - _Red flags:_ Constant low decay; decay stuck at 1.0 with zero updates.
- **Parameter drift (student↔teacher RMS L2)**
  - _Healthy:_ Small but non-zero; decreases and stabilizes in a band.
  - _Red flags:_ ~0 persistently (teacher never updates) or large/increasing (LR too high / EMA too slow/fast).
- **EMA updates count**
  - _Healthy:_ Increments every intended step; matches update cadence.
  - _Red flags:_ Stuck or sporadic increments.

## Optimization sanity
- **Gradient norms (seq-enc, transformer, regressor)**
  - _Healthy:_ Stable band; gentle decay as loss drops.
  - _Red flags:_ Near-zero for long stretches (dead training) or frequent spikes/growth (exploding grads).
- **Parameter norms**
  - _Healthy:_ Slow drift; no runaway growth.
  - _Red flags:_ Rapid growth (instability) or shrinking to ~0 (over-regularization).

## Attention behavior (if available)
- **Attention entropy per layer**
  - _Healthy:_ Mid-range; deeper layers may gradually lower entropy (more focus); early layers remain broader.
  - _Red flags:_ Always maximal (uniform attention) or near-zero from the start (degenerate peaky attention).
- **Attention distance (if positions available)**
  - _Healthy:_ Some locality bias early; deeper layers expand to longer-range dependencies over time.
  - _Red flags:_ Stuck purely local or purely global across all layers.

## Loss breakdown
- **Mean loss per masked token**
  - _Healthy:_ Monotonic decrease early; then plateaus; low variance step-to-step.
  - _Red flags:_ Flat from start; large oscillations; increasing over epochs.
- **Loss by timestep/bucket**
  - _Healthy:_ Coherent pattern (e.g., later steps harder) that narrows with training.
  - _Red flags:_ Random/unstable ordering; some buckets never improve.

## Learned mask vector (`learned_mask`)
- **Mean / Std / Norm**
  - _Healthy:_ Moves away from init; norm increases then stabilizes; std not collapsing to 0.
  - _Red flags:_ Stays near zero (unused token) or norm grows without bound (model over-relies on mask token).

---

## TL;DR Healthy Signals
- Actual mask ≈ scheduled `p`; noise std stable.
- Student–teacher cosine ↑; L2 ↓.
- EMA updates every step; param drift small & stable.
- Grad norms stable; param norms not exploding.
- Attention: early broad, deeper layers gradually sharper/longer-range.
- Mean masked loss ↓; per-bucket losses converge.
- `learned_mask` norm rises then plateaus (not zero, not exploding).
"""