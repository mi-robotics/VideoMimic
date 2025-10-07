'''
File implementing the higher-level policy API (e.g. querying actions and computing losses).
Abstracts away the lower-level architecture details.
'''
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from einops import rearrange, reduce

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from pdp.modules import TransformerForDiffusion
from pdp.modules_v2 import QKVTransformerForDiffusion
from pdp.lora_model import LoraTransformerForDiffusion
from pdp.lora_model import LoraMetaDiffusion
from pdp.modules_v2 import QKVMetaDiffusion

from pdp.utils.normalizer import LinearNormalizer


class DiffusionPolicy(nn.Module):
    def __init__(
        self, 
        model: Union[TransformerForDiffusion, QKVTransformerForDiffusion, LoraTransformerForDiffusion, LoraMetaDiffusion, QKVMetaDiffusion],
        noise_scheduler: DDPMScheduler,
        **kwargs
    ):
        super().__init__()

        self.model = model
        self.obs_dim = self.model.obs_dim
        self.action_dim = self.model.output_dim
        self.T_obs = self.model.T_obs
        self.T_action = self.model.T_action

        self.noise_scheduler = noise_scheduler
        self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        self.normalizer = None # set by set_normalizer

    @property
    def T_range(self):
        return self.T_obs + self.T_action - 1

    def get_optim_groups(self, weight_decay):
        return self.model.get_optim_groups(weight_decay)
    
    # ========= inference  ============
    def conditional_sample(self, cond_data, cond_mask, cond=None,  **kwargs):
        model = self.model
        scheduler = self.noise_scheduler
        trajectory = torch.randn(
            size=cond_data.shape, 
            dtype=cond_data.dtype,
            device=cond_data.device,
        )

        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:
            trajectory[cond_mask] = cond_data[cond_mask]
            model_output, _ = model(trajectory, t, cond,  **kwargs)
    
            # compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(model_output, t, trajectory).prev_sample
        
        trajectory[cond_mask] = cond_data[cond_mask] 
        return trajectory

    def predict_action(self, obs_dict,  **kwargs):
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'obs' in obs_dict
        assert obs_dict['obs'].shape[1:] == (self.T_obs, self.obs_dim)
        nobs = self.normalizer.normalize(obs_dict)['obs']
        B, _, obs_dim = nobs.shape

        # Handle different ways of passing observation
        cond = nobs[:, :self.T_obs]
        shape = (B, self.T_action, self.action_dim)
        cond_data = torch.zeros(size=shape, device=nobs.device, dtype=nobs.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # Run sampling
        nsample = self.conditional_sample(cond_data, cond_mask, cond=cond,  **kwargs)
        
        # Unnormalize prediction and extract action
        naction_pred = nsample[..., :self.action_dim]
        nresult = {'action': naction_pred}
        result = self.normalizer.unnormalize(nresult)
        return result
    
    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer = normalizer

    def get_optimizer(self, weight_decay, learning_rate, betas):
        return self.model.configure_optimizers(
                weight_decay=weight_decay, 
                learning_rate=learning_rate, 
                betas=tuple(betas)
            )
    
    def forward(self, batch):
        return self.compute_loss(batch)
        
    def compute_loss(self, batch):
     
        nbatch = self.normalizer.normalize({
            'obs': batch['obs'],
            'action': batch['action']
        })  
  
        obs = nbatch['obs']
        action = nbatch['action']
        
        cond = obs[:, :self.T_obs]
        start = self.T_obs - 1
        end = start + self.T_action
        trajectory = action[:, start:end]
            
        # generate impainting mask
        condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        B = trajectory.shape[0]
        K = self.noise_scheduler.config.num_train_timesteps

        # Sample a random timestep for each image
        timesteps = torch.randint(0, K, (B,), device=trajectory.device).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask
        
        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        if hasattr(self.model, 'task'):
            if self.model.task == 't2m':
                kwargs = {'caption': batch['caption'], 'caption_emb': batch['caption_emb']}
            elif self.model.task == 'ref':
                kwargs = self.normalizer.normalize({'ref': batch['ref']})
                kwargs['ref'] = kwargs['ref'][:, self.T_obs-1, :]
            elif self.model.task == 'vid_mimic':
                if 'image_emb' in batch:
                    kwargs = {'image_emb': batch['image_emb']}
                else:
                    kwargs = {'image': batch['image']}
            else:
                raise ValueError(f"Unsupported task {self.model.task}")
        else:
            kwargs = {}
            
        pred = self.model(noisy_trajectory, timesteps, cond, **kwargs)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        
        # Only clean up the largest tensors to avoid performance impact
        # del noisy_trajectory, timesteps
        
        return loss, {}



@torch.jit.script
def compute_kernel(x, y):
    x_size = x.shape[0]
    y_size = y.shape[0]
    dim = x.shape[1]

    tiled_x = x.view(x_size, 1, dim).repeat(1, y_size, 1)
    tiled_y = y.view(1, y_size, dim).repeat(x_size, 1, 1)

    return torch.exp(-torch.mean((tiled_x - tiled_y)**2, dim=2)/dim*1.0)

@torch.jit.script
def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return torch.mean(x_kernel) + torch.mean(y_kernel) - 2*torch.mean(xy_kernel)


class MetaDiffusionPolicy(DiffusionPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_dim = self.model.input_dim

    def set_kld_weight(self, kld_weight):
        self.kld_weight = kld_weight

    def predict_action(self, obs_dict,  **kwargs):
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'obs' in obs_dict
        assert 'action' in obs_dict
        assert obs_dict['obs'].shape[1:] == (self.T_obs, self.obs_dim), f'Obs shape mismatch, got {obs_dict["obs"].shape[1:]}, expected {self.T_obs, self.obs_dim}'
        assert obs_dict['action'].shape[1:] == (self.T_obs, self.action_dim), f'Action shape mismatch, got {obs_dict["action"].shape[1:]}, expected {self.T_obs, self.action_dim}'
        ninput = self.normalizer.normalize(obs_dict)
        nobs = ninput['obs']
        naction = ninput['action']

        B, _, obs_dim = nobs.shape
        B, _, action_dim = naction.shape

        # Handle different ways of passing observation
        cond = torch.cat((naction, nobs), dim=-1)
        shape = (B, self.T_action+1, self.action_dim + self.obs_dim)
        cond_data = torch.zeros(size=shape, device=nobs.device, dtype=nobs.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # Run sampling
        nsample = self.conditional_sample(cond_data, cond_mask, cond=cond,  **kwargs)
        
        # Unnormalize prediction and extract action
        naction_pred = nsample[..., :self.action_dim]
        nresult = {'action': naction_pred}
        result = self.normalizer.unnormalize(nresult)
        return result

    def compute_kl_loss(self, mean, logvar):
        return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=-1)

    def compute_mmd_loss(self, mean, logvar):
        # MMD term
     
        true_samples = torch.randn_like(mean.squeeze(), device=mean.device)

        loss_mmd = compute_mmd(true_samples, mean.squeeze())
      
        return loss_mmd

    def compute_loss(self, batch):
        nbatch = self.normalizer.normalize({
            'obs': batch['obs'],
            'action': batch['action']
        })  
  
        obs = nbatch['obs']
        action = nbatch['action']
        
        
        trajectory = torch.cat((action, obs), dim=-1)

        if isinstance(self.model, LoraMetaDiffusion):
            #we create an overlap, the first state to be predicted is provided in the context
            cond = trajectory[:, :self.T_obs]
            trajectory = trajectory[:, self.T_obs-1:, :] # Dropping the last s,a pair, so our prediction len remains T_action
        elif isinstance(self.model, QKVMetaDiffusion):
            
            cond = trajectory[:, :self.T_obs]
            trajectory = trajectory[:, self.T_obs-1:, :] 
        else: 
            raise ValueError(f"Unsupported model {self.model}")
        # generate impainting mask
        condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        B = trajectory.shape[0]
        K = self.noise_scheduler.config.num_train_timesteps

        # Sample a random timestep for each image
        timesteps = torch.randint(0, K, (B,), device=trajectory.device).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask
        
        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        if hasattr(self.model, 'task'):
            if self.model.task == 't2m':
                kwargs = {'caption': batch['caption'], 'caption_emb': batch['caption_emb']}
            elif self.model.task == 'ref':
                kwargs = self.normalizer.normalize({'ref': batch['ref']})
                kwargs['ref'] = kwargs['ref'][:, self.T_obs-1, :]
            elif self.model.task == 'vid_mimic':
                if 'image_emb' in batch:
                    kwargs = {'image_emb': batch['image_emb']}
                else:
                    kwargs = {'image': batch['image']}
            elif self.model.task == 'pref_comp':
                kwargs = {}
            else:
                raise ValueError(f"Unsupported task {self.model.task}")
        else:
            kwargs = {}
            
        if isinstance(self.model, LoraMetaDiffusion):
            pred, info_dict = self.model(noisy_trajectory, timesteps, cond, **kwargs)
        elif isinstance(self.model, QKVMetaDiffusion):
            pred, info_dict = self.model(trajectory, noisy_trajectory, timesteps, cond, **kwargs)
        else: 
            raise ValueError(f"Unsupported model {self.model}")
        
        

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss_action = F.mse_loss(pred[:, :, :self.action_dim], target[:, :, :self.action_dim], reduction='none')
        loss_action = loss_action * loss_mask[:, :, :self.action_dim].type(loss_action.dtype)
        loss_action = reduce(loss_action, 'b ... -> b (...)', 'mean')
        loss_action = loss_action.mean()


        loss_obs = F.mse_loss(pred[:, :, self.action_dim:], target[:, :, self.action_dim:], reduction='none')
        loss_obs = loss_obs * loss_mask[:, :, self.action_dim:].type(loss_obs.dtype)
        loss_obs = reduce(loss_obs, 'b ... -> b (...)', 'mean')
        loss_obs = loss_obs.mean()

        loss_dict = {}
        loss_dict['loss_action'] = loss_action.detach().item()
        loss_dict['loss_obs'] = loss_obs.detach().item()



        if isinstance(self.model, LoraMetaDiffusion):
            loss = loss_action + loss_obs 

            self.use_latent_guidance = False 
            if self.use_latent_guidance:
                with torch.no_grad():
                    target_latent = self.model.get_target_latent(trajectory, timesteps, cond=cond, **kwargs)
                latent_loss = F.mse_loss(info_dict['cls_emb'], target_latent, reduction='mean')
                loss_dict['latent_loss'] = latent_loss.detach().item()
                loss = loss + 10*latent_loss.mean()

            if self.model.learn_latent and self.model.lora_model.is_variational and self.model.lora_model.mmd_weight > 0:
                
                mmd_loss = self.compute_mmd_loss(info_dict['cls_emb'], None)
                loss = loss / self.noise_scheduler.config.num_train_timesteps
                loss = loss + self.model.lora_model.mmd_weight * mmd_loss
                loss_dict['mmd_loss'] = mmd_loss.detach().item()
              

        elif isinstance(self.model, QKVMetaDiffusion):
            loss = loss_action + loss_obs
            if self.model.learn_latent and self.model.is_variational :

                if self.model.mmd_weight > 0:
                    mmd_loss = self.compute_mmd_loss(info_dict['cls_mean'], info_dict['cls_logvar'])

                    loss = loss / self.noise_scheduler.config.num_train_timesteps
                    loss = loss + self.model.mmd_weight * mmd_loss

                    loss_dict['mmd_loss'] = mmd_loss.detach().item()
                
                else:
                    # loss = loss / self.noise_scheduler.config.num_train_timesteps
                    kl = self.compute_kl_loss(info_dict['cls_mean'], info_dict['cls_logvar']).mean()
                    kl_loss = self.model.kl_beta * self.kld_weight * kl
                    # kl_loss = self.model.kl_beta * kl

                    loss = loss + kl_loss
                    
                    loss_dict['kl_loss'] = kl_loss.detach().item()
                    loss_dict['kl'] = kl.detach().item()
      

        else: 
            raise ValueError(f"Unsupported model {self.model}")
        

        
        return loss, loss_dict
        