import torch
import torch.nn as nn

from pdp.world_model.rssm import RSSM

import numpy as np
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F

from pdp.utils.normalizer import LinearNormalizer

class DenseModel(nn.Module):
    def __init__(
            self, 
            output_shape,
            input_size, 
            info,
        ):
        """
        :param output_shape: tuple containing shape of expected output
        :param input_size: size of input features
        :param info: dict containing num of hidden layers, size of hidden layers, activation function, output distribution etc.
        """
        super().__init__()
        self._output_shape = output_shape
        self._input_size = input_size
        self._layers = info['layers']
        self._node_size = info['node_size']
        self.activation = eval(info['activation'])
        self.dist = info['dist']
        self.model = self.build_model()

    def build_model(self):
        model = [nn.Linear(self._input_size, self._node_size)]
        model += [self.activation()]
        for i in range(self._layers-1):
            model += [nn.Linear(self._node_size, self._node_size)]
            model += [self.activation()]
        model += [nn.Linear(self._node_size, int(np.prod(self._output_shape)))]
        return nn.Sequential(*model)

    def forward(self, input):
        dist_inputs = self.model(input)

        if self.dist == 'normal':
            return td.independent.Independent(td.Normal(dist_inputs, 1), len(self._output_shape))
        if self.dist == None:
            return dist_inputs

        raise NotImplementedError(self._dist)

class DreamerNet(nn.Module):
    def __init__(self, 
        obs_shape, 
        action_size, 
        rssm_type, 
        rssm_info, 
        embedding_size, 
        rssm_node_size, 
        obs_encoder, 
        obs_decoder,

        kl_loss_weight,
        kl_info,

        T_range,
        
        ):

        super().__init__()

        self.kl_loss_weight = kl_loss_weight
        self.kl_info = kl_info
        self.T_range = T_range
        obs_shape = obs_shape
        action_size = action_size
        deter_size = rssm_info['deter_size']

        if rssm_type == 'continuous':
            stoch_size = rssm_info['stoch_size']

        elif rssm_type == 'discrete':
            category_size = rssm_info['category_size']
            class_size = rssm_info['class_size']
            stoch_size = category_size*class_size


        modelstate_size = stoch_size + deter_size 
    
        self.RSSM = RSSM(action_size, rssm_node_size, embedding_size, rssm_type, rssm_info)

        self.ObsEncoder = DenseModel((embedding_size,), int(np.prod(obs_shape)), obs_encoder)
        self.ObsDecoder = DenseModel(obs_shape, modelstate_size, obs_decoder)
     

    def get_optim_groups(self, weight_decay):
 
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, DenseModel, RSSM)
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

    def forward(self, batch):
        return self.compute_loss(batch)

    def compute_loss(self, batch):

        nbatch = self.normalizer.normalize({
            'obs': batch['obs'],
            'action': batch['action']
        })  

        obs = nbatch['obs'][:, 1:].permute(1,0,2) # sequence first
        actions = nbatch['action'][:,:-1].permute(1,0,2) # prev actions

        model_loss, kl_loss, obs_loss, prior_dist, post_dist, posterior \
            = self.representation_loss(obs, actions)

        return model_loss, {
            'kl_loss': kl_loss.detach().item(),
            'obs_loss': obs_loss.detach().item(),
        }


    def representation_loss(self, obs, actions):
        T, B, _ = obs.size()
      

        embed = self.ObsEncoder(obs)                                         #t to t+seq_len   
        prev_rssm_state = self.RSSM._init_rssm_state(B, device=obs.device)   
        prior, posterior = self.RSSM.rollout_observation(T, embed, actions, prev_rssm_state)
        post_modelstate = self.RSSM.get_model_state(posterior)               #t to t+seq_len   
        obs_dist = self.ObsDecoder(post_modelstate[:-1])                     #t to t+seq_len-1  
   
        obs_loss = self._obs_loss(obs_dist, obs[:-1])
        prior_dist, post_dist, div = self._kl_loss(prior, posterior)

        model_loss = self.kl_loss_weight * div  + obs_loss 
        return model_loss, div, obs_loss, prior_dist, post_dist, posterior


    def _obs_loss(self, obs_dist, obs):
        obs_loss = -torch.mean(obs_dist.log_prob(obs))
        return obs_loss

    
    def _kl_loss(self, prior, posterior):
        prior_dist = self.RSSM.get_dist(prior)
        post_dist = self.RSSM.get_dist(posterior)
        if self.kl_info['use_kl_balance']:
            alpha = self.kl_info['kl_balance_scale']
            kl_lhs = torch.mean(torch.distributions.kl.kl_divergence(self.RSSM.get_dist(self.RSSM.rssm_detach(posterior)), prior_dist))
            kl_rhs = torch.mean(torch.distributions.kl.kl_divergence(post_dist, self.RSSM.get_dist(self.RSSM.rssm_detach(prior))))
            if self.kl_info['use_free_nats']:
                free_nats = self.kl_info['free_nats']
                kl_lhs = torch.max(kl_lhs,kl_lhs.new_full(kl_lhs.size(), free_nats))
                kl_rhs = torch.max(kl_rhs,kl_rhs.new_full(kl_rhs.size(), free_nats))
            kl_loss = alpha*kl_lhs + (1-alpha)*kl_rhs

        else: 
            kl_loss = torch.mean(torch.distributions.kl.kl_divergence(post_dist, prior_dist))
            if self.kl_info['use_free_nats']:
                free_nats = self.kl_info['free_nats']
                kl_loss = torch.max(kl_loss, kl_loss.new_full(kl_loss.size(), free_nats))
        return prior_dist, post_dist, kl_loss




class DeltaWorldModel(nn.Module):
    def __init__(self,
        obs_shape,
        action_size,
        rssm_type,
        rssm_info,
        embedding_size,
        rssm_node_size,
        obs_encoder,
        obs_decoder,
        loss_type,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.DeltaPredictor = DenseModel(obs_shape, action_size+int(np.prod(obs_shape)), obs_encoder)
        return 

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer = normalizer

    def compute_loss(self, batch):

        nbatch = self.normalizer.normalize({
            'obs': batch['obs'],
            'action': batch['action']
        })  

        if self.loss_type == 'single_step':
            obs = nbatch['obs'][:, :-1] # sequence first
            actions = nbatch['action'][:,:-1]# prev actions
            next_obs = nbatch['obs'][:, 1:]
            delta = next_obs - obs
            delta_pred = self.DeltaPredictor(torch.cat((actions, obs), dim=-1))
            model_loss = F.mse_loss(delta_pred, delta)
            loss_dict = {}

        elif self.loss_type == 'multi_step':
            T  = obs.shape[1]-1

            next_obs_targets = nbatch['obs'][:, 1:]
            next_obs_preds = []
            obs = nbatch['obs'][:, 0] #bs, obs_dim
            actions = nbatch['action'][:, 0] #bs, action_dim
            for t in range(T):
                delta_pred = self.DeltaPredictor(torch.cat((actions, obs), dim=-1))
                next_obs = delta_pred + obs
                next_obs_preds.append(next_obs)

                obs = next_obs
                actions = nbatch['action'][:, t+1] #bs, action_dim

            next_obs_preds = torch.cat(next_obs_preds, dim=1)
            model_loss = F.mse_loss(next_obs_preds, next_obs_targets)
            loss_dict = {}

      

        return model_loss, loss_dict