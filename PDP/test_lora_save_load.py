#!/usr/bin/env python3
"""
Test script for LoRA save/load functionality.

This script demonstrates how to use the LoRA save/load methods
in line with the workspace parameter saving and loading pattern.
"""

import torch
import tempfile
import pathlib
from pdp.lora_model import LoraTransformerForDiffusion
from pdp.modules_v2 import QKVTransformerForDiffusion
from pdp.utils.lora_utils import (
    save_lora_checkpoint,
    load_lora_checkpoint,
    create_lora_model_from_checkpoint,
    get_lora_model_info,
    print_lora_model_summary
)


def test_lora_model_save_load():
    """Test basic LoRA model save/load functionality."""
    print("=" * 60)
    print("Testing LoRA Model Save/Load Functionality")
    print("=" * 60)
    
    # Create a base model
    print("1. Creating base model...")
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
    print("2. Creating LoRA model...")
    lora_model = LoraTransformerForDiffusion(
        input_dim=69,
        output_dim=69,
        obs_dim=357,
        emb_dim=256,
        T_obs=4,
        T_action=2,
        n_encoder_layers=2,
        n_decoder_layers=4,
        n_head=4,
        p_drop_emb=0.0,
        p_drop_attn=0.1,
        obs_type='ref',
        causal_attn=True,
        past_action_visible=False,
        task="t2m",
        lora_encoder_units=[512, 512],
        cond_mechanism="add",
        teacher_ckpt_path=None,  # Would normally load from checkpoint
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_module_list=['q_proj', 'k_proj', 'out_proj'],
        apply_to="both"
    )
    
    # Print model summary
    print("3. Model Summary:")
    print_lora_model_summary(lora_model)
    
    # Test 1: Save only LoRA adapters (most efficient)
    print("\n4. Testing LoRA adapters save/load...")
    with tempfile.TemporaryDirectory() as temp_dir:
        adapters_path = pathlib.Path(temp_dir) / "lora_adapters.pth"
        
        # Save LoRA adapters
        lora_model.save_lora_adapters(adapters_path)
        
        # Create a new LoRA model and load adapters
        lora_model_2 = LoraTransformerForDiffusion(
            input_dim=69,
            output_dim=69,
            obs_dim=357,
            emb_dim=256,
            T_obs=4,
            T_action=2,
            n_encoder_layers=2,
            n_decoder_layers=4,
            n_head=4,
            p_drop_emb=0.0,
            p_drop_attn=0.1,
            obs_type='ref',
            causal_attn=True,
            past_action_visible=False,
            task="t2m",
            lora_encoder_units=[512, 512],
            cond_mechanism="add",
            teacher_ckpt_path=None,
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_module_list=['q_proj', 'k_proj', 'out_proj'],
            apply_to="both"
        )
        
        # Load LoRA adapters
        lora_model_2.load_lora_adapters(adapters_path)
        print("✓ LoRA adapters save/load successful")
    
    # Test 2: Save full checkpoint
    print("\n5. Testing full checkpoint save/load...")
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = pathlib.Path(temp_dir) / "lora_full_checkpoint.pth"
        
        # Save full checkpoint
        lora_model.save_full_checkpoint(checkpoint_path)
        
        # Load full checkpoint
        lora_model_3 = LoraTransformerForDiffusion.from_checkpoint(
            checkpoint_path=checkpoint_path,
            input_dim=69,
            output_dim=69,
            obs_dim=357,
            emb_dim=256,
            T_obs=4,
            T_action=2,
            n_encoder_layers=2,
            n_decoder_layers=4,
            n_head=4,
            p_drop_emb=0.0,
            p_drop_attn=0.1,
            obs_type='ref',
            causal_attn=True,
            past_action_visible=False,
            task="t2m",
            lora_encoder_units=[512, 512],
            cond_mechanism="add",
            teacher_ckpt_path=None,
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_module_list=['q_proj', 'k_proj', 'out_proj'],
            apply_to="both"
        )
        print("✓ Full checkpoint save/load successful")
    
    # Test 3: Utility functions
    print("\n6. Testing utility functions...")
    with tempfile.TemporaryDirectory() as temp_dir:
        utility_path = pathlib.Path(temp_dir) / "utility_checkpoint.pth"
        
        # Create optimizer
        optimizer = torch.optim.AdamW(lora_model.get_optim_groups(weight_decay=0.001))
        
        # Save using utility function
        save_lora_checkpoint(
            model=lora_model,
            path=utility_path,
            optimizer=optimizer,
            epoch=1,
            global_step=100,
            metadata={'description': 'Test LoRA checkpoint'}
        )
        
        # Load using utility function
        payload = load_lora_checkpoint(
            path=utility_path,
            model=lora_model,
            optimizer=optimizer,
            load_optimizer=True
        )
        
        print(f"✓ Utility functions successful. Metadata: {payload.get('pickles', {}).get('metadata', {})}")
    
    print("\n" + "=" * 60)
    print("All LoRA save/load tests completed successfully!")
    print("=" * 60)


def test_workspace_integration():
    """Test LoRA integration with workspace."""
    print("\n" + "=" * 60)
    print("Testing Workspace Integration")
    print("=" * 60)
    
    # This would require a full workspace setup, so we'll just show the pattern
    print("Workspace integration pattern:")
    print("""
    # In your workspace, you can now use:
    
    # Check if model is LoRA
    if workspace.is_lora_model():
        # Save LoRA checkpoint (optimized)
        workspace.save_lora_checkpoint(tag='latest')
        
        # Load LoRA checkpoint
        workspace.load_lora_checkpoint(tag='latest')
    else:
        # Use standard checkpoint methods
        workspace.save_checkpoint(tag='latest')
        workspace.load_checkpoint(tag='latest')
    
    # The workspace will automatically use LoRA checkpoints during training
    # when save_checkpoint_every is reached.
    """)
    
    print("✓ Workspace integration pattern documented")


if __name__ == "__main__":
    try:
        test_lora_model_save_load()
        test_workspace_integration()
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


