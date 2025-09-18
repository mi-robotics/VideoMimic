"""
Example usage of LoRA model saving and loading in the PDP framework.

This script demonstrates how to:
1. Create a LoRA model
2. Save LoRA adapters only
3. Save full LoRA checkpoints
4. Load LoRA models from checkpoints
5. Use LoRA models with the workspace
"""

import torch
import hydra
from omegaconf import OmegaConf
from pathlib import Path

# Import your framework components
from pdp.lora_model import LoraTransformerForDiffusion
from pdp.modules_v2 import QKVTransformerForDiffusion
from pdp.utils.lora_utils import (
    save_lora_checkpoint, 
    load_lora_checkpoint,
    create_lora_model_from_checkpoint,
    merge_lora_weights,
    save_merged_model
)


def example_1_basic_lora_save_load():
    """Example 1: Basic LoRA model creation, saving, and loading."""
    print("=== Example 1: Basic LoRA Save/Load ===")
    
    # Create a base model
    base_model = QKVTransformerForDiffusion(
        obs_type='ref',
        causal_attn=True,
        past_action_visible=False,
        obs_dim=357,
        input_dim=69,
        output_dim=69,
        emb_dim=256,
        T_obs=4,
        T_action=2,
        n_encoder_layers=2,
        n_decoder_layers=4,
        n_head=4,
        p_drop_attn=0.1,
        p_drop_emb=0.0,
    )
    
    # Create LoRA model
    lora_model = LoraTransformerForDiffusion(
        base_model=base_model,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=['q_proj', 'k_proj', 'out_proj'],
        apply_to='both'
    )
    
    print(f"Created LoRA model with {sum(p.numel() for p in lora_model.parameters() if p.requires_grad)} trainable parameters")
    
    # Method 1: Save only LoRA adapters (most efficient)
    lora_model.save_lora_adapters('lora_adapters_only.pth')
    
    # Method 2: Save full checkpoint (compatible with framework)
    lora_model.save_full_checkpoint('lora_full_checkpoint.pth')
    
    # Load LoRA adapters
    lora_model.load_lora_adapters('lora_adapters_only.pth')
    
    # Load full checkpoint
    lora_model.load_full_checkpoint('lora_full_checkpoint.pth')
    
    print("✓ Basic save/load completed successfully")


def example_2_workspace_integration():
    """Example 2: Using LoRA models with the workspace."""
    print("\n=== Example 2: Workspace Integration ===")
    
    # Create a configuration for LoRA model
    cfg = OmegaConf.create({
        '_target_': 'pdp.workspace.DiffusionPolicyWorkspace',
        'policy': {
            '_target_': 'pdp.policy.DiffusionPolicy',
            'model': {
                '_target_': 'pdp.lora_model.LoraTransformerForDiffusion',
                'base_model': {
                    '_target_': 'pdp.modules_v2.QKVTransformerForDiffusion',
                    'obs_type': 'ref',
                    'causal_attn': True,
                    'past_action_visible': False,
                    'obs_dim': 357,
                    'input_dim': 69,
                    'output_dim': 69,
                    'emb_dim': 256,
                    'T_obs': 4,
                    'T_action': 2,
                    'n_encoder_layers': 2,
                    'n_decoder_layers': 4,
                    'n_head': 4,
                    'p_drop_attn': 0.1,
                    'p_drop_emb': 0.0,
                },
                'lora_r': 8,
                'lora_alpha': 16,
                'lora_dropout': 0.1,
                'target_modules': ['q_proj', 'k_proj', 'out_proj'],
                'apply_to': 'both'
            },
            'noise_scheduler': {
                '_target_': 'diffusers.schedulers.scheduling_ddpm.DDPMScheduler',
                'beta_end': 0.02,
                'beta_schedule': 'squaredcos_cap_v2',
                'beta_start': 0.0001,
                'clip_sample': True,
                'num_train_timesteps': 10,
                'prediction_type': 'sample',
                'variance_type': 'fixed_small',
            }
        },
        'training': {
            'device': 'cpu',
            'debug': True,
            'logging': False,
            'use_ema': True,
            'save_checkpoint_every': 1,
            'num_epochs': 1,
            'seed': 42
        },
        'optimizer': {
            'betas': [0.9, 0.95],
            'lr': 0.0001,
            'weight_decay': 0.001
        }
    })
    
    # Create workspace with LoRA model
    workspace = hydra.utils.instantiate(cfg)
    
    print(f"Workspace created with LoRA model: {workspace.is_lora_model()}")
    
    # Save LoRA checkpoint using workspace
    workspace.save_lora_checkpoint('workspace_lora_checkpoint.pth')
    
    # Load LoRA checkpoint using workspace
    workspace.load_lora_checkpoint('workspace_lora_checkpoint.pth')
    
    print("✓ Workspace integration completed successfully")


def example_3_utility_functions():
    """Example 3: Using utility functions for LoRA checkpoint handling."""
    print("\n=== Example 3: Utility Functions ===")
    
    # Create a base model
    base_model = QKVTransformerForDiffusion(
        obs_type='ref',
        causal_attn=True,
        past_action_visible=False,
        obs_dim=357,
        input_dim=69,
        output_dim=69,
        emb_dim=256,
        T_obs=4,
        T_action=2,
        n_encoder_layers=2,
        n_decoder_layers=4,
        n_head=4,
        p_drop_attn=0.1,
        p_drop_emb=0.0,
    )
    
    # Create LoRA model
    lora_model = LoraTransformerForDiffusion(
        base_model=base_model,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=['q_proj', 'k_proj', 'out_proj'],
        apply_to='both'
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(lora_model.get_optim_groups(weight_decay=0.001))
    
    # Save checkpoint using utility function
    save_lora_checkpoint(
        model=lora_model,
        path='utility_checkpoint.pth',
        optimizer=optimizer,
        epoch=1,
        global_step=100,
        metadata={'description': 'Example LoRA checkpoint'}
    )
    
    # Load checkpoint using utility function
    payload = load_lora_checkpoint(
        path='utility_checkpoint.pth',
        model=lora_model,
        optimizer=optimizer,
        load_optimizer=True
    )
    
    print(f"Loaded checkpoint with metadata: {payload.get('metadata', {})}")
    
    # Create LoRA model from checkpoint
    new_lora_model = create_lora_model_from_checkpoint(
        checkpoint_path='utility_checkpoint.pth',
        base_model_class=QKVTransformerForDiffusion,
        base_model_kwargs={
            'obs_type': 'ref',
            'causal_attn': True,
            'past_action_visible': False,
            'obs_dim': 357,
            'input_dim': 69,
            'output_dim': 69,
            'emb_dim': 256,
            'T_obs': 4,
            'T_action': 2,
            'n_encoder_layers': 2,
            'n_decoder_layers': 4,
            'n_head': 4,
            'p_drop_attn': 0.1,
            'p_drop_emb': 0.0,
        }
    )
    
    print("✓ Utility functions completed successfully")


def example_4_merge_lora_weights():
    """Example 4: Merging LoRA weights into base model."""
    print("\n=== Example 4: Merge LoRA Weights ===")
    
    # Create a base model
    base_model = QKVTransformerForDiffusion(
        obs_type='ref',
        causal_attn=True,
        past_action_visible=False,
        obs_dim=357,
        input_dim=69,
        output_dim=69,
        emb_dim=256,
        T_obs=4,
        T_action=2,
        n_encoder_layers=2,
        n_decoder_layers=4,
        n_head=4,
        p_drop_attn=0.1,
        p_drop_emb=0.0,
    )
    
    # Create LoRA model
    lora_model = LoraTransformerForDiffusion(
        base_model=base_model,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=['q_proj', 'k_proj', 'out_proj'],
        apply_to='both'
    )
    
    # Merge LoRA weights into base model
    merged_model = merge_lora_weights(lora_model, alpha=1.0)
    
    # Save merged model
    save_merged_model(merged_model, 'merged_model.pth')
    
    print("✓ LoRA weights merged and saved successfully")


def example_5_class_method_loading():
    """Example 5: Using class method to create LoRA model from checkpoint."""
    print("\n=== Example 5: Class Method Loading ===")
    
    # First, create and save a LoRA model
    base_model = QKVTransformerForDiffusion(
        obs_type='ref',
        causal_attn=True,
        past_action_visible=False,
        obs_dim=357,
        input_dim=69,
        output_dim=69,
        emb_dim=256,
        T_obs=4,
        T_action=2,
        n_encoder_layers=2,
        n_decoder_layers=4,
        n_head=4,
        p_drop_attn=0.1,
        p_drop_emb=0.0,
    )
    
    lora_model = LoraTransformerForDiffusion(
        base_model=base_model,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=['q_proj', 'k_proj', 'out_proj'],
        apply_to='both'
    )
    
    # Save checkpoint
    lora_model.save_full_checkpoint('class_method_checkpoint.pth')
    
    # Load using class method
    loaded_lora_model = LoraTransformerForDiffusion.from_checkpoint(
        checkpoint_path='class_method_checkpoint.pth',
        base_model_path=None  # Base model is included in checkpoint
    )
    
    print("✓ Class method loading completed successfully")


def cleanup_example_files():
    """Clean up example files."""
    files_to_remove = [
        'lora_adapters_only.pth',
        'lora_full_checkpoint.pth',
        'workspace_lora_checkpoint.pth',
        'utility_checkpoint.pth',
        'merged_model.pth',
        'class_method_checkpoint.pth'
    ]
    
    for file_path in files_to_remove:
        if Path(file_path).exists():
            Path(file_path).unlink()
            print(f"Removed: {file_path}")


if __name__ == "__main__":
    print("LoRA Model Save/Load Examples")
    print("=" * 50)
    
    try:
        example_1_basic_lora_save_load()
        example_2_workspace_integration()
        example_3_utility_functions()
        example_4_merge_lora_weights()
        example_5_class_method_loading()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\nCleaning up example files...")
        cleanup_example_files()
        print("Cleanup completed.")
