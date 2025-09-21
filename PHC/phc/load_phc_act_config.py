#!/usr/bin/env python3
"""
Script to load the PHC Humanoid environment using Hydra configuration loading.

This script uses Hydra to load the actual configuration files from the PHC codebase,
matching the configuration used in create_phc_act_dataset.py.
"""


# import sys
# # Add the PHC directory to Python path
# sys.path.append('/home/mcarroll/Documents/cd-2/VideoMimic/PHC')
# sys.path.append('/home/mcarroll/Documents/cd-2/VideoMimic/PHC/phc')
import glob
import os
import sys
import pdb
import os.path as osp
os.environ["OMP_NUM_THREADS"] = "1"

sys.path.append(os.getcwd())


from phc.utils.config import set_np_formatting, set_seed
from phc.utils.parse_task import parse_task_simple
from phc.utils.flags import flags
from isaacgym import gymapi
import hydra
from omegaconf import DictConfig, OmegaConf

import os
import argparse
import numpy as np
import torch
from easydict import EasyDict

def load_config_with_hydra(additional_overrides=None):
    """Load configuration using Hydra, matching the PHC ACT setup."""
    
    # Initialize Hydra with the PHC config path
    config_path = './data/cfg'
    
    # Base overrides from PHC ACT command
    base_overrides = [
        "learning=im_mcp",
        "env=env_im_getup_mcp", 
        "robot=smpl_humanoid",
        "robot.freeze_hand=True",
        "robot.box_body=False",
        "env.z_activation=relu",
        "env.obs_v=7",
        "env.add_action_noise=False",
        "env.action_noise_std=0.0",
        "env.collect_dataset=False",
        "test=True",
        "im_eval=True",
        "headless=False",
        "exp_name=phc_kp_mcp_iccv"
    ]
    
    # Add additional overrides if provided
    if additional_overrides:
        base_overrides.extend(additional_overrides)
    
    # Compose the configuration using Hydra
    with hydra.initialize(config_path=config_path, version_base=None):
        cfg_hydra = hydra.compose(
            config_name="config",
            overrides=base_overrides
        )
    
    # Convert to EasyDict for compatibility
    cfg = EasyDict(OmegaConf.to_container(cfg_hydra, resolve=True))
    
    return cfg


def parse_sim_params(cfg):
    """Parse simulation parameters."""
    sim_params = gymapi.SimParams()
    sim_params.dt = eval(cfg.sim.physx.step_dt)
    sim_params.num_client_threads = cfg.sim.slices
    
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = 4
    sim_params.physx.use_gpu = cfg.sim.pipeline in ["gpu"]
    sim_params.physx.num_subscenes = cfg.sim.subscenes
    sim_params.physx.max_gpu_contact_pairs = 16 * 1024 * 1024

    sim_params.use_gpu_pipeline = cfg.sim.pipeline in ["gpu"]
    sim_params.physx.use_gpu = cfg.sim.pipeline in ["gpu"]
    
    return sim_params


def load_humanoid_env_with_phc_act_config(num_envs=512, additional_overrides=None):
    """Load the Humanoid environment with PHC ACT configuration using Hydra."""
    
    # Set up formatting and seed
    set_np_formatting()
    set_seed(42, torch_deterministic=False)
    
    # Prepare additional overrides
    overrides = []
    if num_envs:
        overrides.append(f"env.num_envs={num_envs}")
    if additional_overrides:
        overrides.extend(additional_overrides)
    
    # Load configuration using Hydra
    cfg = load_config_with_hydra(overrides if overrides else None)
    cfg.env.enableEarlyTermination = False
    
    # Set up flags based on PHC ACT configuration
    flags.debug = False
    flags.follow = False
    flags.fixed = False
    flags.divide_group = False
    flags.no_collision_check = False
    flags.fixed_path = False
    flags.real_path = False
    flags.small_terrain = False
    flags.show_traj = True
    flags.server_mode = False
    flags.slow = False
    flags.real_traj = False
    flags.im_eval = cfg.im_eval
    flags.no_virtual_display = True
    flags.render_o3d = False
    flags.test = cfg.test
    flags.add_proj = False
    flags.has_eval = True
    flags.trigger_input = False
    
    # Set up simulation parameters
    sim_params = parse_sim_params(cfg)
    
    # Create arguments for task parsing
    args = EasyDict({
        "task": 'Humanoid',
        "device_id": cfg.device_id,
        "rl_device": cfg.rl_device,
        "physics_engine": gymapi.SIM_PHYSX,
        "headless": cfg.headless,
        "device": cfg.device,
    })
    
    # Create training config (minimal for environment creation)
    cfg_train = {
        'params': {
            'config': {
                'clip_observations': np.inf,
                'num_actors': cfg.env.num_envs
            }
        }
    }
    
    # Parse and create the task and environment
    task, env = parse_task_simple(args, cfg, cfg_train, sim_params)
    
    return task, env, cfg


def main():
    """Main function to demonstrate PHC ACT configuration loading."""
    
    parser = argparse.ArgumentParser(
        description='Load Humanoid environment with PHC ACT configuration using Hydra',
        epilog="""
Examples:
  # Basic usage with default PHC ACT configuration
  python load_phc_act_config.py
  
  # With additional Hydra overrides
  python load_phc_act_config.py --hydra_overrides env.num_envs=256 env.episode_length=500
  
  # Single environment for testing
  python load_phc_act_config.py --num_envs 1 --headless False
        """
    )
    parser.add_argument('--num_envs', type=int, default=512,
                       help='Number of parallel environments')
    parser.add_argument('--headless', action='store_true', default=True,
                       help='Run in headless mode')
    parser.add_argument('--num_steps', type=int, default=1000,
                       help='Number of simulation steps to run')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--hydra_overrides', type=str, nargs='*', default=None,
                       help='Additional Hydra overrides (e.g., env.num_envs=256)')
    
    args = parser.parse_args()
    
    print("Loading PHC Humanoid Environment with PHC ACT Configuration...")
    print(f"Configuration loaded using Hydra from PHC config files")
    import time
    try:
        # Load the environment with PHC ACT config
        task, env, cfg = load_humanoid_env_with_phc_act_config(
            num_envs=args.num_envs,
            additional_overrides=args.hydra_overrides
        )
        
        print(f"\n✅ Environment loaded successfully!")
        print(f"   Task: {cfg.env.task}")
        # print(f"   Learning: {cfg.learning.type}")
        print(f"   Number of environments: {env.num_envs}")
        print(f"   Number of actions: {env.num_actions}")
        print(f"   Number of observations: {env.num_obs}")
        print(f"   Device: {cfg.device}")
        print(f"   Test mode: {cfg.test}")
        print(f"   IM eval: {cfg.im_eval}")
        print(f"   Exp name: {cfg.exp_name}")
        
        # Test basic functionality
        print(f"\n🧪 Testing basic functionality...")
        
        # Reset environment
        obs = env.reset()
        print(f"   ✅ Reset successful - observation shape: {obs.shape}")
        
        # Take a random action
        actions = torch.rand(env.num_envs, env.num_actions, device='cpu') * 2 - 1
        obs, rewards, dones, infos = env.step(actions)
        print(f"   ✅ Step successful - mean reward: {rewards.mean().item():.4f}")
        
        # Run a short simulation
        print(f"\n🏃 Running simulation for {args.num_steps} steps...")
        for step in range(args.num_steps):
            actions = torch.rand(env.num_envs, env.num_actions, device='cpu') * 2 - 1
            obs, rewards, dones, infos = env.step(actions)
            
            if step % 100 == 0:
                print(f"   Step {step}: Mean reward = {rewards.mean().item():.4f}, "
                      f"Dones = {dones.sum().item()}")
            
            if dones.any():
                done_indices = torch.where(dones)[0]
                obs = env.reset(done_indices)

            time.sleep(0.1)
        
        print(f"\n🎉 Simulation completed successfully!")
        
        # Clean up
        env.close()
        print(f"   ✅ Environment closed successfully.")
        
    except Exception as e:
        print(f"❌ Error loading environment: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
