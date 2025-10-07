from ast import Pass
import time
import torch
import torch.nn as nn
from pdp.modules_v2 import QKVTransformerForDiffusion, QKVAttention, QKVCrossAttention, QKVMetaDiffusion
import torch
from PIL import Image
import requests
from transformers import AutoImageProcessor, AutoModel
import cv2
import torch
from transformers.image_utils import load_image

from peft import get_peft_model, LoraConfig, TaskType
import dill
import clip



def load_clip( clip_version='ViT-B/32', device='cpu'):
    clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                            jit=False)  # Must set jit=False for training
    clip.model.convert_weights(
        clip_model)  # Actually this line is unnecessary since clip by default already on float16

    # Freeze CLIP weights
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    clip_model = clip_model.to(device)

    clip_func = lambda x: clip_encode_text(clip_model, x)

    return clip_func, 512


def clip_encode_text(clip_model, raw_text):
    # raw_text - list (batch_size length) of strings with input text prompts
   
    max_text_len = 20  # Specific hardcoding for humanml dataset
    if max_text_len is not None:
        default_context_length = 77
        context_length = max_text_len + 2 # start_token + 20 + end_token
        assert context_length < default_context_length
        texts = clip.tokenize(raw_text, context_length=context_length, truncate=True)# [bs, context_length] # if n_tokens > context_length -> will truncate
        # print('texts', texts.shape)
        zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
        texts = torch.cat([texts, zero_pad], dim=1).cuda()
        # print('texts after pad', texts.shape, texts)
    else:
        texts = clip.tokenize(raw_text, truncate=True).cuda()# [bs, context_length] # if n_tokens > 77 -> will truncate
    return clip_model.encode_text(texts).float() # [bs, embed dim]



def load_dinov2(device='cuda'):
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base')
    model = model.to(device)
    dino_fn = lambda x: dino_encode_image(x, processor, model)
    return dino_fn, 768

def dino_encode_image(images, processor, model):
    inputs = processor(images=images, return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    return last_hidden_states[:, 0, :]



class LoraTransformerForDiffusion(nn.Module):
    """
    Wraps a QKVTransformerForDiffusion model with PEFT LoRA adapters.
    """
    def __init__(
        self,
        
        input_dim, output_dim, obs_dim, emb_dim, T_obs, T_action,
        n_encoder_layers=4, n_decoder_layers=4, n_head=4,
        p_drop_emb=0.1, p_drop_attn=0.1,
        obs_type=None, causal_attn=False, past_action_visible=False,

        task: str = "t2m",
        lora_encoder_units: list = [512, 512],
        cond_mechanism: str = "add",
        teacher_ckpt_path: str = None,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        cond_mask_prob: float = 0.0,
        target_module_list: list = ['q_proj', 'k_proj', 'out_proj'],
        apply_to: str = "both",  # "encoder", "decoder", or "both"
        **kwargs
    ):
        super().__init__()
        if get_peft_model is None:
            raise ImportError("peft is not installed. Please install peft to use LoraTransformerForDiffusion.")

        #TODO this is a really bad pattern need to fix, make it load form config
        base_model = QKVTransformerForDiffusion(
            input_dim=input_dim,
            output_dim=output_dim,
            obs_dim=obs_dim,
            emb_dim=emb_dim,
            T_obs=T_obs,
            T_action=T_action,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
            n_head=n_head,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            obs_type=obs_type,
            causal_attn=causal_attn,
            past_action_visible=past_action_visible)
        self.init_base_model(base_model, teacher_ckpt_path)

        self.obs_dim = base_model.obs_dim
        self.input_dim = base_model.input_dim
        self.output_dim = base_model.output_dim
        self.emb_dim = base_model.emb_dim
        self.T_obs = base_model.T_obs
        self.T_action = base_model.T_action
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        # Disable gradients for all parameters in the base model
        for param in base_model.parameters():
            param.requires_grad = False

        self.apply_to = apply_to
        self.cond_mask_prob = cond_mask_prob

        # Validate apply_to parameter
        if apply_to not in ["encoder", "decoder", "both"]:
            raise ValueError("apply_to must be one of: 'encoder', 'decoder', 'both'")

        # Default target_modules: all Linear layers in the model
        self.target_module_list = target_module_list
        target_modules = self._find_linear_module_names(base_model, apply_to)

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            # Better suited for encoder-decoder architecture
        )

        self.lora_model = get_peft_model(base_model, lora_config)

        self.task = task
        self.cond_mechanism = cond_mechanism
        self.lora_encoder_units = lora_encoder_units

        if self.cond_mechanism not in ["add", 'cat', 'film', 'film-attn']:
            raise NotImplementedError(f"Condition mechanism {self.cond_mechanism} not supported")
        if self.cond_mechanism == "film":
            self.film1 = torch.nn.Sequential(
                nn.Mish(),
                nn.Linear(emb_dim*3, emb_dim*4, bias=True),
                nn.Mish(),
                nn.Linear(emb_dim*4, emb_dim*4, bias=True),
                nn.Mish(),
                nn.Linear(emb_dim*4, 2*emb_dim*self.T_obs, bias=True)
            )
        if self.cond_mechanism == "film-attn":
            self.encoder_films = torch.nn.Sequential(
                nn.Mish(),
                nn.Linear(emb_dim*2, emb_dim*4, bias=True),
                nn.Mish(),
                nn.Linear(emb_dim*4, emb_dim*4, bias=True),
                nn.Mish(),
                nn.Linear(emb_dim*4, self.n_encoder_layers*4, bias=True)
            )
            self.decoder_films = torch.nn.Sequential(
                nn.Mish(),
                nn.Linear(emb_dim*2, emb_dim*4, bias=True),
                nn.Mish(),
                nn.Linear(emb_dim*4, emb_dim*4, bias=True),
                nn.Mish(),
                nn.Linear(emb_dim*4, self.n_decoder_layers*6, bias=True)
            )

        if self.task == "t2m":
            # load clip
            self.clip_fn, self.lora_backbone_dim = load_clip(device='cuda')
        elif self.task == "ref":
            self.lora_backbone_dim = 438
        elif self.task == "vid_mimic":
            self.dino_fn, self.lora_backbone_dim = load_dinov2(device='cuda')
            self.lora_backbone_dim = self.lora_backbone_dim*2
        else:
            raise NotImplementedError(f"Task {self.task} not supported")
       
        #Initialize the lora encoder
        lora_encoder_layers = []
        in_units = self.lora_backbone_dim
        for unit in self.lora_encoder_units:
            lora_encoder_layers.append(torch.nn.Linear(in_units, unit))
            lora_encoder_layers.append(torch.nn.GELU())
            lora_encoder_layers.append(torch.nn.LayerNorm(unit))
            lora_encoder_layers.append(torch.nn.Dropout(0.1))
            in_units = unit
        lora_encoder_layers.append(torch.nn.Linear(in_units, emb_dim))
        self.lora_encoder = torch.nn.Sequential(*lora_encoder_layers)
    

        return 


    def init_base_model(self, base_model, ckpt_path):
        payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
        state_dict_loadable = {}

        for k, v in payload['state_dicts']['model'].items():
            if 'model.' in k:
                k_ = k.replace('model.', '')
                state_dict_loadable[k_] = v

        base_model.load_state_dict(state_dict_loadable)


    def _find_linear_module_names(self, model, apply_to="both"):
        """
        Recursively find nn.Linear module names based on which part of the model to target.
        
        Args:
            model: The QKVTransformerForDiffusion model
            apply_to: "encoder", "decoder", or "both"
        """
        if apply_to == "none":
            return []
        linear_names = set()
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Determine which part of the model this module belongs to
                is_encoder = any(keyword in name.lower() for keyword in [
                    'encoder'
                ])
                is_decoder = any(keyword in name.lower() for keyword in [
                    'decoder'
                ])
                
                # Apply filtering based on apply_to parameter
                should_include = False
                if apply_to == "encoder" and is_encoder:
                    should_include = True
                elif apply_to == "decoder" and is_decoder:
                    should_include = True
                elif apply_to == "both" and (is_encoder or is_decoder):
                    should_include = True
                
                if should_include:
                    for target in self.target_module_list:
                        if target in name:
                            linear_names.add(name)
                  
        print('Implimenting LoRA on the following modules:')
        for name in linear_names:
            print(f"  - {name}")
        print('===================================')
        return list(linear_names)


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
        no_decay.add("lora_model.base_model.model.pos_emb")
        no_decay.add("lora_model.base_model.model.cond_pos_emb")

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
        #TODO this pattern is nasty and brittle to base model changes
        """
        sample: (B, T_action, input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B, T_obs, cond_dim)
        """
        assert torch.is_tensor(timestep)
        assert cond.shape == (sample.shape[0], self.lora_model.T_obs, self.lora_model.obs_dim)
        assert sample.shape == (cond.shape[0], self.lora_model.T_action, self.lora_model.input_dim)
        if len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)

        self.lora_model.base_model.training = self.training
        
        if self.task == "t2m":
            assert 'caption' in kwargs
            caption = kwargs['caption']

            if 'caption_emb' in kwargs:
                text_emb = kwargs['caption_emb']
            else:
                text_emb = self.clip_fn(caption)

            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

            lora_cond_emb = self.lora_encoder(text_emb) # [bs, emb dim]

        elif self.task == "ref":
            lora_cond_emb = self.lora_encoder(kwargs['ref'])

        elif self.task == "vid_mimic":
            assert 'image' in kwargs or 'image_emb' in kwargs
            
            if 'image_emb' in kwargs:
                image_emb = kwargs['image_emb']
            else:
                image = kwargs['image']
                image_emb = self.dino_fn(image) # [bs, 2, emb_dim]

            # Normalize and flatten with explicit memory management
            image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
            image_emb = image_emb.flatten(1,2)
            
            # Process through LoRA encoder
            lora_cond_emb = self.lora_encoder(image_emb) # [bs, emb dim]
            
            # Clean up intermediate tensors to free memory
            del image_emb

        else:
            raise ValueError(f"Task {self.task} not supported")

        

 
        # Encoder for conditioning
        timesteps = timestep.expand(sample.shape[0])
        time_emb = self.lora_model.time_emb(timesteps).unsqueeze(1)
        # (B, 1, obs_dim)

        cond_emb = self.lora_model.cond_obs_emb(cond)
        
        if self.cond_mechanism == "add":
            cond_emb = torch.cat([time_emb + lora_cond_emb.unsqueeze(1), cond_emb], dim=1)
        elif self.cond_mechanism == "film": 
         
            text_mask = self.mask_batch(lora_cond_emb.shape[0], batch_mask_prob=self.cond_mask_prob, training=self.training)
  
            scale_shifts = self.film1(torch.hstack([
                time_emb.squeeze(1), 
                lora_cond_emb,
                cond_emb[:,-1].squeeze(1)])).view(-1, 2, self.T_obs, self.emb_dim)

            text_scale, text_shift = torch.chunk(scale_shifts, 2, dim=1)

            text_scale = text_scale.squeeze(1).clone()
            text_shift = text_shift.squeeze(1).clone()
            text_scale[text_mask] = text_scale[text_mask] * 0 
            text_shift[text_mask] = text_shift[text_mask] * 0

            
            text_scale = text_scale.squeeze(1)
            text_shift = text_shift.squeeze(1)
            cond_emb = ( 1 + text_scale) * cond_emb + text_shift 
            
            cond_emb = torch.cat([time_emb, cond_emb], dim=1)

        elif self.cond_mechanism == "film-attn":
            encoder_scale_shifts = self.encoder_films(torch.hstack([
                time_emb.squeeze(1), 
                lora_cond_emb])).view(-1, self.n_encoder_layers, 4)
            decoder_scale_shifts = self.decoder_films(torch.hstack([
                time_emb.squeeze(1), 
                lora_cond_emb])).view(-1, self.n_decoder_layers, 6)
          
            kwargs['encoder_scale_shifts'] = encoder_scale_shifts
            kwargs['decoder_scale_shifts'] = decoder_scale_shifts
        else:
            cond_emb = torch.cat([time_emb, cond_emb], dim=1)

        tc = cond_emb.shape[1]
        cond_pos_emb = self.lora_model.cond_pos_emb[:, :tc, :]
        x = self.lora_model.drop(cond_emb + cond_pos_emb)

        if self.cond_mechanism == "cat":
            x = torch.cat([x, lora_cond_emb.unsqueeze(1)], dim=1)

        x = self.lora_model.encoder(x, **kwargs)
        memory = x 
        # (B, T_cond, obs_dim)

        # Decoder for action prediction
        input_emb = self.lora_model.input_emb(sample)

        t = sample.shape[1]
        pos_emb = self.lora_model.pos_emb[:, :t, :]
        x = self.lora_model.drop(input_emb + pos_emb)
        x = self.lora_model.decoder(
            x=x,
            memory=memory,
            tgt_mask=self.lora_model.mask,
            memory_mask=None,
            **kwargs
        )
        # (B, T, obs_dim)
        # NOTE: We don't need a memory mask because the conditioning is always on past information
        
        x = self.lora_model.ln_f(x)
        x = self.lora_model.head(x)
        # (B, T, output_dim)
        
        # Clean up intermediate tensors to prevent memory accumulation
        # del memory, input_emb, pos_emb, cond_emb, time_emb, cond_pos_emb
        
        return x

    
    def mask_batch(self, batch_size, batch_mask_prob=0.0, training=True):
        if batch_mask_prob > 0.0 and training:
            mask = torch.rand(batch_size) < batch_mask_prob
            return mask
        return torch.zeros(batch_size, dtype=torch.bool)
    

    def get_targeted_modules(self):
        """
        Return the list of modules that have LoRA adapters applied.
        Useful for debugging and understanding what's being fine-tuned.
        """
        return self._find_linear_module_names(self.lora_model.base_model, self.apply_to)
     
 
    




class LoraMetaDiffusion(LoraTransformerForDiffusion):
    def __init__(
        self,
        
        input_dim, output_dim, obs_dim, emb_dim, T_obs, T_action,
        n_encoder_layers=4, n_decoder_layers=4, n_head=4,
        p_drop_emb=0.1, p_drop_attn=0.1,
        obs_type=None, causal_attn=False, past_action_visible=False,

        task: str = "t2m",
        lora_encoder_units: list = [512, 512],
        hist_encoder_units: list = [512, 512],
        cond_mechanism: str = None,
        teacher_ckpt_path: str = None,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        cond_mask_prob: float = 0.0,
        target_module_list: list = ['q_proj', 'k_proj', 'out_proj'],
        apply_to: str = "both",  # "encoder", "decoder", or "both"
        is_variational: bool = False,
        latent_size: int = 64,
        kl_beta: float = 1,
        mmd_weight: float = 0.0,
        normalize_latent: bool = False,
        is_enc_past_visible: bool = False,
        condition_mechanism: str = None,
        **kwargs
    ):
        nn.Module.__init__(self)
        if get_peft_model is None:
            raise ImportError("peft is not installed. Please install peft to use LoraTransformerForDiffusion.")

        #TODO this is a really bad pattern need to fix, make it load form config
        base_model = QKVMetaDiffusion(
            input_dim=input_dim,
            output_dim=output_dim,
            obs_dim=obs_dim,
            emb_dim=emb_dim,
            T_obs=T_obs,
            T_action=T_action,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
            n_head=n_head,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            obs_type=obs_type,
            causal_attn=causal_attn,
            past_action_visible=past_action_visible, 
            is_variational=is_variational,
            latent_size=latent_size,
            kl_beta=kl_beta,
            mmd_weight=mmd_weight,
            normalize_latent=normalize_latent,
            is_enc_past_visible=is_enc_past_visible,
            condition_mechanism=condition_mechanism)
        self.init_base_model(base_model, teacher_ckpt_path)

        self.obs_dim = base_model.obs_dim
        self.input_dim = base_model.input_dim
        self.output_dim = base_model.output_dim
        self.emb_dim = base_model.emb_dim
        self.T_obs = base_model.T_obs
        self.T_action = base_model.T_action
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        # Disable gradients for all parameters in the base model
        for param in base_model.parameters():
            param.requires_grad = False

        self.apply_to = apply_to
        self.cond_mask_prob = cond_mask_prob

        # Validate apply_to parameter
        if apply_to not in ["encoder", "decoder", "both", "none"]:
            raise ValueError("apply_to must be one of: 'encoder', 'decoder', 'both', 'none'")

        # Default target_modules: all Linear layers in the model
        self.target_module_list = target_module_list
        target_modules = self._find_linear_module_names(base_model, apply_to)

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            # Better suited for encoder-decoder architecture
        )

        self.lora_model = get_peft_model(base_model, lora_config)

        self.task = task
        self.cond_mechanism = cond_mechanism
        self.lora_encoder_units = lora_encoder_units
        self.hist_encoder_units = hist_encoder_units
        self.is_enc_past_visible = is_enc_past_visible
        assert self.cond_mechanism is None, "Cond mechanism not supported for Meta Diffusion"
     
        if self.task == "pref_comp":
            self.lora_backbone_dim = 0
        elif self.task == "t2m":
            # load clip
            self.clip_fn, self.lora_backbone_dim = load_clip(device='cuda')
        elif self.task == "ref":
            self.lora_backbone_dim = 438
        elif self.task == "vid_mimic":
            self.dino_fn, self.lora_backbone_dim = load_dinov2(device='cuda')
            self.lora_backbone_dim = self.lora_backbone_dim*2
        else:
            raise NotImplementedError(f"Task {self.task} not supported")

        
       
        #Initialize the lora encoder
        lora_encoder_layers = []
        in_units = self.lora_backbone_dim + self.emb_dim # emb dim because we need to learn from the context embedding
        for unit in self.lora_encoder_units:
            lora_encoder_layers.append(torch.nn.Linear(in_units, unit))
            lora_encoder_layers.append(torch.nn.GELU())
            lora_encoder_layers.append(torch.nn.LayerNorm(unit))
            lora_encoder_layers.append(torch.nn.Dropout(0.1))
            in_units = unit
        lora_encoder_layers.append(torch.nn.Linear(in_units, self.lora_model.latent_size))
        self.lora_encoder = torch.nn.Sequential(*lora_encoder_layers)

        hist_enc_layers = []
        in_units = self.emb_dim # emb dim because we need to learn from the context embedding
        for unit in self.hist_encoder_units:
            hist_enc_layers.append(torch.nn.Linear(in_units, unit))
            hist_enc_layers.append(torch.nn.GELU())
            hist_enc_layers.append(torch.nn.LayerNorm(unit))
            hist_enc_layers.append(torch.nn.Dropout(0.1))
            in_units = unit
        hist_enc_layers.append(torch.nn.Linear(in_units, self.emb_dim))
        self.hist_encoder = torch.nn.Sequential(*hist_enc_layers)




        return

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
        no_decay.add("lora_model.base_model.model.pos_emb_enc")
        no_decay.add("lora_model.base_model.model.pos_emb_dec")
        no_decay.add("lora_model.base_model.model.cls_token")

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


    def get_target_latent(self, sample_clean, timestep, cond=None, **kwargs):
        """
        sample: (B, T_action, input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B, T_obs, cond_dim)
        """
        assert torch.is_tensor(timestep)
        assert cond.shape[1:] == (self.T_obs, self.input_dim+self.obs_dim)
        assert sample_clean.shape[1:] == (self.T_action+1, self.input_dim+self.obs_dim)
        if len(timestep.shape) == 0:
            timestep = timestep[None].to(sample_clean.device)



        actions = sample_clean[:, 1:, :self.input_dim]
        obs = sample_clean[:, 1:, self.input_dim:]
        actions_cond = cond[:, :, :self.input_dim]
        obs_cond = cond[:, :, self.input_dim:]

        actions = torch.cat((actions_cond, actions), dim=1)
        obs = torch.cat((obs_cond, obs), dim=1)
        # Decoder for action prediction
        action_emb = self.lora_model.action_emb_enc(actions) # [bs, seq, emb_dim]
        obs_emb = self.lora_model.obs_emb_enc(obs) # [bs, seq, emb_dim]

        assert actions.shape[1] == self.T_action + self.T_obs
        assert obs.shape[1] == self.T_action + self.T_obs

        x_in = torch.cat((obs_emb.unsqueeze(-2), action_emb.unsqueeze(-2)), dim=-2).flatten(1,2) # [bs, seq*2, emb_dim] --> s, a, s, a 

        x_in = x_in 
        x_in = torch.cat((
            self.lora_model.cls_token.repeat(x_in.shape[0], 1, 1),
            x_in,
        ), dim=1)
        # (B, T_cond, obs_dim)

        t = x_in.shape[1]
        pos_emb = self.lora_model.pos_emb_enc[:, :t, :]
        x_in = self.lora_model.drop(x_in + pos_emb)
        x_in = self.lora_model.encoder(x_in, **kwargs)

        cls_emb = x_in[:, 0:1, :]

        if self.lora_model.is_variational:
            cls_emb = self.lora_model.cls_token_proj(cls_emb)
            cls_mean, cls_logvar = cls_emb.chunk(2, dim=-1)
            cls_emb = self.lora_model.reparameterize(cls_mean, cls_logvar)
        else:
            cls_emb = self.lora_model.cls_token_proj(cls_emb)
        ############# Denoiser #############

        if self.lora_model.normalize_latent:
            cls_emb = cls_emb / cls_emb.norm(dim=-1, keepdim=True)
        
        return cls_emb


    def forward(self, sample, timestep, cond=None, **kwargs):
        #TODO this pattern is nasty and brittle to base model changes
        """
        sample: (B, T_action, input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B, T_obs, cond_dim)
        """
        self.lora_model.base_model.training = self.training
        assert torch.is_tensor(timestep)
        assert cond.shape[1:] == (self.T_obs, self.input_dim+self.obs_dim)
        assert sample.shape[1:] == (self.T_action+1, self.input_dim+self.obs_dim), f'Sample shape mismatch, got {sample.shape[1:]}, expected {self.T_action+1, self.input_dim+self.obs_dim}'
        if len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)

        # Encoder for conditioning
        timesteps = timestep.expand(sample.shape[0])
        time_emb = self.lora_model.time_emb(timesteps).unsqueeze(1)

        # Note we want the condition to end with the most recent observations
        # Eg we can pass the state ass [(s,a),..., (s,a)] and just drop the last state
        
        ### Step 1 --- Aquire the context and prediction vectors

        ## Context --- 
        actions = cond[:, :, :self.input_dim]
        obs = cond[:, :, self.input_dim:]
        action_emb = self.lora_model.action_emb_enc(actions) # [bs, seq, emb_dim]
        obs_emb = self.lora_model.obs_emb_enc(obs) # [bs, seq, emb_dim]
        x_in_context = torch.cat((obs_emb.unsqueeze(-2), action_emb.unsqueeze(-2)), dim=-2).flatten(1,2) 
        # x_in_context = x_in_context #+ time_emb
        x_in_context = x_in_context[:, :-1, :] # Dropping the last action
        x_in_context = torch.cat((
            self.lora_model.cls_token.repeat(x_in_context.shape[0], 1, 1),
            x_in_context,
        ), dim=1)

        ## Preds --- 
        actions = sample[:, :, :self.input_dim]
        obs = sample[:, 1:, self.input_dim:]
        actions_cond = cond[:, :-1, :self.input_dim]
        obs_cond = cond[:, :, self.input_dim:]

        actions = torch.cat((actions_cond, actions), dim=1)
        obs = torch.cat((obs_cond, obs), dim=1)

        assert actions.shape[1] == self.T_action + self.T_obs
        assert obs.shape[1] == self.T_action + self.T_obs
        # Decoder for action prediction
        action_emb = self.lora_model.action_emb_dec(actions) # [bs, seq, emb_dim]
        obs_emb = self.lora_model.obs_emb_dec(obs) # [bs, seq, emb_dim]

        x_in = torch.cat((obs_emb.unsqueeze(-2), action_emb.unsqueeze(-2)), dim=-2).flatten(1,2) # [bs, seq*2, emb_dim] --> s, a, s, a 
        x_in_preds = x_in + time_emb




        ### Step 2 --- Aquire an embedding of the context
        t = x_in_context.shape[1]
        pos_emb = self.lora_model.pos_emb_enc[:, :t, :]
        x_in_context = self.lora_model.drop(x_in_context + pos_emb)
        x_in_context = self.lora_model.encoder(x_in_context, **kwargs)
        cls_emb = x_in_context[:, 0:1, :]

        # if self.lora_model.is_variational:
        #     cls_emb = self.lora_model.cls_token_proj(cls_emb)
        #     cls_mean, cls_logvar = cls_emb.chunk(2, dim=-1)
        #     cls_emb = self.lora_model.reparameterize(cls_mean, cls_logvar)
        #     # cls_emb = cls_mean

        cls_emb = cls_emb[:, 0, :]
        cls_emb = self.hist_encoder(cls_emb)
        
        ### Step 3 --- Transform context and condition into cls_embedding 
        
        if self.task == "t2m":
            assert 'caption' in kwargs
            caption = kwargs['caption']

            if 'caption_emb' in kwargs:
                text_emb = kwargs['caption_emb']
            else:
                text_emb = self.clip_fn(caption)

            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

            lora_in = torch.cat((cls_emb, text_emb), dim=-1)

            lora_cond_emb = self.lora_encoder(lora_in) # [bs, emb dim]

        elif self.task == "ref":
            lora_in = torch.cat((cls_emb, kwargs['ref']), dim=-1)
            lora_cond_emb = self.lora_encoder(lora_in)

        elif self.task == "vid_mimic":
            assert 'image' in kwargs or 'image_emb' in kwargs
            
            if 'image_emb' in kwargs:
                image_emb = kwargs['image_emb']
            else:
                image = kwargs['image']
                image_emb = self.dino_fn(image) # [bs, 2, emb_dim]

            # Normalize and flatten with explicit memory management
            image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
            image_emb = image_emb.flatten(1,2)
            lora_in = torch.cat((cls_emb, image_emb), dim=-1)
            # Process through LoRA encoder
            lora_cond_emb = self.lora_encoder(lora_in) # [bs, emb dim]
            
            # Clean up intermediate tensors to free memory
            del image_emb

        elif self.task == "pref_comp":
            lora_in = cls_emb
            lora_cond_emb = self.lora_encoder(lora_in)

        else:
            raise ValueError(f"Task {self.task} not supported")


        if self.lora_model.normalize_latent:
            lora_cond_emb = lora_cond_emb / lora_cond_emb.norm(dim=-1, keepdim=True)
    
        # Step 4 -- Decoder the actions state trajectory
        if self.lora_model.condition_mechanism == 'cat':
            x_in_preds = torch.cat((
                self.lora_model.latent_up_proj(lora_cond_emb).unsqueeze(1),
                x_in_preds,
            ), dim=1)

            memory = None
        elif self.lora_model.condition_mechanism == 'cross_attn':
            memory = self.lora_model.latent_up_proj(lora_cond_emb).unsqueeze(1)
        else:
            raise ValueError(f"Unsupported condition mechanism: {self.lora_model.condition_mechanism}")



        t = x_in_preds.shape[1]
        pos_emb = self.lora_model.pos_emb_dec[:, :t, :]
        x_in_preds = self.lora_model.drop(x_in_preds + pos_emb)
        x = self.lora_model.decoder(
            x=x_in_preds,
            tgt_mask=self.lora_model.mask,
            memory_mask=None,
            memory=memory,
            **kwargs
        )

       
        if self.lora_model.condition_mechanism == 'cat':
            x = x[:, 1:, :]
        # (B, T, obs_dim)
        x_obs = x[:, 0::2, :]
        x_obs = x_obs[:, self.T_obs-1:, :]
        x_action = x[:, 1::2, :]
        x_action = x_action[:, self.T_obs-1:, :]
     

        x_action = self.lora_model.head_action(self.lora_model.ln_action(x_action))
        x_obs = self.lora_model.head_obs(self.lora_model.ln_obs(x_obs))

        x_out = torch.cat((x_action, x_obs), dim=-1)
        return x_out, { 'cls_emb': lora_cond_emb }