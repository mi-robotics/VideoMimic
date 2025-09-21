"""
Utility functions for LoRA model checkpoint handling in the PDP framework.
"""

import torch
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union


def save_lora_checkpoint(
    model, 
    path: Union[str, Path], 
    optimizer=None, 
    scheduler=None, 
    epoch: int = 0, 
    global_step: int = 0,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Save a LoRA model checkpoint compatible with the PDP framework.
    
    Args:
        model: LoRA model instance (LoraTransformerForDiffusion)
        path: Path to save the checkpoint
        optimizer: Optimizer state (optional)
        scheduler: Learning rate scheduler state (optional)
        epoch: Current epoch number
        global_step: Current global step
        metadata: Additional metadata to save
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create checkpoint payload compatible with PDP framework
    payload = {
        'cfg': getattr(model, 'cfg', None),
        'state_dicts': {},
        'pickles': {},
        'epoch': epoch,
        'global_step': global_step,
        'metadata': metadata or {}
    }
    
    # Save model state dict
    if hasattr(model, 'lora_model'):
        # For LoRA models, save the full model state dict
        payload['state_dicts']['model'] = model.lora_model.state_dict()
        payload['state_dicts']['ema_model'] = model.lora_model.state_dict()  # For compatibility
        
        # Save LoRA configuration
        payload['pickles']['lora_config'] = {
            'lora_r': model.lora_model.peft_config['default'].r,
            'lora_alpha': model.lora_model.peft_config['default'].lora_alpha,
            'lora_dropout': model.lora_model.peft_config['default'].lora_dropout,
            'target_modules': model.lora_model.peft_config['default'].target_modules,
            'apply_to': getattr(model, 'apply_to', 'both'),
        }
    else:
        # For regular models
        payload['state_dicts']['model'] = model.state_dict()
        payload['state_dicts']['ema_model'] = model.state_dict()
    
    # Save optimizer state
    if optimizer is not None:
        payload['state_dicts']['optimizer'] = optimizer.state_dict()
    
    # Save scheduler state
    if scheduler is not None:
        payload['state_dicts']['scheduler'] = scheduler.state_dict()
    
    # Save checkpoint
    torch.save(payload, path)
    print(f"LoRA checkpoint saved to: {path}")


def load_lora_checkpoint(
    path: Union[str, Path], 
    model=None, 
    optimizer=None, 
    scheduler=None,
    load_optimizer: bool = True,
    load_scheduler: bool = True,
    strict: bool = False
):
    """
    Load a LoRA model checkpoint compatible with the PDP framework.
    
    Args:
        path: Path to the checkpoint file
        model: Model instance to load weights into (optional)
        optimizer: Optimizer instance to load state into (optional)
        scheduler: Scheduler instance to load state into (optional)
        load_optimizer: Whether to load optimizer state
        load_scheduler: Whether to load scheduler state
        strict: Whether to use strict loading
        
    Returns:
        Dict containing the loaded checkpoint data
    """
    path = Path(path)
    payload = torch.load(path, map_location='cpu')
    
    print(f"Loading LoRA checkpoint from: {path}")
    print(f"Available keys: {list(payload.get('state_dicts', {}).keys())}")
    
    # Load model state dict
    if model is not None and 'model' in payload['state_dicts']:
        if hasattr(model, 'lora_model'):
            # For LoRA models
            missing_keys, unexpected_keys = model.lora_model.load_state_dict(
                payload['state_dicts']['model'], strict=strict
            )
        else:
            # For regular models
            missing_keys, unexpected_keys = model.load_state_dict(
                payload['state_dicts']['model'], strict=strict
            )
        
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
    
    # Load optimizer state
    if optimizer is not None and load_optimizer and 'optimizer' in payload['state_dicts']:
        optimizer.load_state_dict(payload['state_dicts']['optimizer'])
        print("Optimizer state loaded")
    
    # Load scheduler state
    if scheduler is not None and load_scheduler and 'scheduler' in payload['state_dicts']:
        scheduler.load_state_dict(payload['state_dicts']['scheduler'])
        print("Scheduler state loaded")
    
    return payload


def create_lora_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    base_model_class,
    base_model_kwargs: Dict[str, Any],
    lora_kwargs: Optional[Dict[str, Any]] = None
):
    """
    Create a LoRA model instance from a saved checkpoint.
    
    Args:
        checkpoint_path: Path to the LoRA checkpoint
        base_model_class: Class of the base model
        base_model_kwargs: Arguments for creating the base model
        lora_kwargs: Additional arguments for LoRA configuration
        
    Returns:
        LoRA model instance
    """
    from pdp.lora_model import LoraTransformerForDiffusion
    
    payload = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract LoRA configuration
    lora_config = payload.get('pickles', {}).get('lora_config', {})
    
    # Create base model
    base_model = base_model_class(**base_model_kwargs)
    
    # Create LoRA model
    lora_kwargs = lora_kwargs or {}
    lora_model = LoraTransformerForDiffusion(
        base_model=base_model,
        lora_r=lora_config.get('lora_r', 8),
        lora_alpha=lora_config.get('lora_alpha', 16),
        lora_dropout=lora_config.get('lora_dropout', 0.1),
        target_modules=lora_config.get('target_modules', ['q_proj', 'k_proj', 'out_proj']),
        apply_to=lora_config.get('apply_to', 'both'),
        **lora_kwargs
    )
    
    # Load the weights
    if 'model' in payload['state_dicts']:
        lora_model.lora_model.load_state_dict(payload['state_dicts']['model'])
    
    return lora_model


def merge_lora_weights(model, alpha: float = 1.0):
    """
    Merge LoRA weights into the base model weights.
    This creates a single model without LoRA adapters.
    
    Args:
        model: LoRA model instance
        alpha: Scaling factor for LoRA weights
        
    Returns:
        Model with merged weights
    """
    if not hasattr(model, 'lora_model'):
        raise ValueError("Model must be a LoRA model")
    
    # Use PEFT's built-in merge functionality
    merged_model = model.lora_model.merge_and_unload()
    return merged_model


def save_merged_model(model, path: Union[str, Path]):
    """
    Save a model with merged LoRA weights (no LoRA adapters).
    
    Args:
        model: Model with merged weights
        path: Path to save the model
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_type': 'merged_lora'
    }
    
    torch.save(checkpoint, path)
    print(f"Merged model saved to: {path}")


def compare_lora_configs(config1: Dict[str, Any], config2: Dict[str, Any]) -> bool:
    """
    Compare two LoRA configurations for compatibility.
    
    Args:
        config1: First LoRA configuration
        config2: Second LoRA configuration
        
    Returns:
        True if configurations are compatible, False otherwise
    """
    required_keys = ['lora_r', 'lora_alpha', 'lora_dropout', 'target_modules']
    
    for key in required_keys:
        if key not in config1 or key not in config2:
            return False
        if config1[key] != config2[key]:
            return False
    
    return True
