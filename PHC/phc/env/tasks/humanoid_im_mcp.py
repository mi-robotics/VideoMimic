import time
import torch
import phc.env.tasks.humanoid_im as humanoid_im
from phc.env.tasks.humanoid_im import compute_pdp_obs

from isaacgym.torch_utils import *
from phc.utils.flags import flags
from rl_games.algos_torch import torch_ext
import torch.nn as nn
from phc.learning.pnn import PNN
from collections import deque
from phc.learning.network_loader import load_mcp_mlp, load_pnn
from phc.learning.mlp import MLP
import collections

def load_pdp_policy(policy_path):
    import dill
    import hydra
    payload = torch.load(open(policy_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cfg.dataset.cache_data = False
    cfg.training.device = cfg.training.device if torch.cuda.is_available() else 'cpu'
    workspace = hydra.utils.get_class(cfg._target_)(cfg)
    workspace.load_payload(payload)

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.eval()
    return policy, cfg

class HumanoidImMCP(humanoid_im.HumanoidIm):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.num_prim = cfg["env"].get("num_prim", 3)
        self.discrete_mcp = cfg["env"].get("discrete_moe", False)
        self.has_pnn = cfg["env"].get("has_pnn", False)
        self.has_lateral = cfg["env"].get("has_lateral", False)
        self.z_activation = cfg["env"].get("z_activation", "relu")
        self.pdp_policy_path = cfg["env"].get("pdp_policy_path", None)

        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)


        if self.pdp_policy_path:
            self.pdp_policy, self.pdp_cfg = load_pdp_policy(self.pdp_policy_path)
            pdp_obs = compute_pdp_obs(
                self._rigid_body_pos[:], 
                self._rigid_body_rot[:], 
                self._rigid_body_vel[:],
                self._rigid_body_ang_vel[:]
            ) #[n envs, pdp_obs_size]
  
            self.pdp_hist = pdp_obs.unsqueeze(1).repeat(1, self.pdp_cfg.policy.model.T_obs, 1) 
    
       
        if self.has_pnn:
            assert (len(self.models_path) == 1)
            pnn_ck = torch_ext.load_checkpoint(self.models_path[0])
            self.pnn = load_pnn(pnn_ck, num_prim = self.num_prim, has_lateral = self.has_lateral, activation = self.z_activation, device = self.device)
            self.running_mean, self.running_var = pnn_ck['running_mean_std']['running_mean'], pnn_ck['running_mean_std']['running_var']
        
        if self.mlp_bypass:
    
            self.mlp_model = MLP(input_dim = self.num_obs, output_dim=self.num_dof, units = [2048, 1024, 512], activation = "silu")
            checkpoint = torch.load(self.mlp_model_path)
            if 'model_state_dict' in checkpoint:
                self.mlp_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.mlp_model.load_state_dict(checkpoint)
            self.mlp_model.to(self.device)

        self.fps = deque(maxlen=90)
        
        return

    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)
        self._num_actions = self.num_prim
        return

    def get_task_obs_size_detail(self):
        task_obs_detail = super().get_task_obs_size_detail()
        task_obs_detail['num_prim'] = self.num_prim
        return task_obs_detail


    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if self.pdp_policy_path:
            self.pdp_hist[env_ids] = self.pdp_obs_buff[env_ids].unsqueeze(1).repeat(1, self.pdp_cfg.policy.model.T_obs, 1)
        return

    def step(self, weights):

        # if self.dr_randomizations.get('actions', None):
        #     actions = self.dr_randomizations['actions']['noise_lambda'](actions)
        # if flags.server_mode:
            # t_s = time.time()
        
        with torch.no_grad():
            # Apply trained Model.
            curr_obs = ((self.obs_buf - self.running_mean.float().to(self.device)) / torch.sqrt(self.running_var.float().to(self.device) + 1e-05))
            curr_obs = torch.clamp(curr_obs, min=-5.0, max=5.0)
            
            if self.pdp_policy_path:
          
                self.pdp_hist[:, :-1, :] = self.pdp_hist[:, 1:, :]
                self.pdp_hist[:, -1, :] = self.pdp_obs_buff[:]
                action_dict = self.pdp_policy.predict_action({
                    'obs': self.pdp_hist.clone()
                })
                actions = action_dict['action'][:, 0, :]
            else:
                if self.discrete_mcp:
                    max_idx = torch.argmax(weights, dim=1)
                    weights = torch.nn.functional.one_hot(max_idx, num_classes=self.num_prim).float()
                
                if self.has_pnn:
                    _, actions = self.pnn(curr_obs)
                    x_all = torch.stack(actions, dim=1)
                    actions = torch.sum(weights[:, :, None] * x_all, dim=1)
                    if flags.debug:
                        print("\npnn output actions \n", actions[0][0:8])
                else:
                    x_all = torch.stack([net(curr_obs) for net in self.actors], dim=1)
                    actions = torch.sum(weights[:, :, None] * x_all, dim=1)
                    if flags.debug:
                        print("\nnot pnn output actions \n", actions[0][0:8])

                if self.mlp_bypass:
                    actions = self.mlp_model(curr_obs)
                    if flags.debug:
                        print("\nmlp input observations \n", curr_obs[0][0:8])
                        print("\nmlp actions\n", actions[0][0:8])
            
            #import pdb; pdb.set_trace()
            # print(weights)
     
        # actions = x_all[:, 3]  # Debugging
        # apply actions
        self.pre_physics_step(actions)

        # step physics and render each frame
        self._physics_step()

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()
        # if flags.server_mode:
        #     dt = time.time() - t_s
        #     print(f'\r {1/dt:.2f} fps', end='')
            
        # dt = time.time() - t_s
        # self.fps.append(1/dt)
        # print(f'\r {np.mean(self.fps):.2f} fps', end='')
        
        # import time
        # time.sleep(0.01)
        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

  
