"""
Utility functions for LoRA model saving, loading, and manipulation.

This module provides helper functions for working with LoRA models in the PDP framework,
including checkpoint management, weight merging, and model creation utilities.
"""

import torch
import dill
import pathlib
from typing import Dict, Any, Optional, Union, List
from omegaconf import OmegaConf


def save_lora_checkpoint(
    model,
    path: Union[str, pathlib.Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    global_step: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs
) -> str:
    """
    Save a LoRA checkpoint with additional metadata.
    
    Args:
        model: LoRA model to save
        path: Path to save the checkpoint
        optimizer: Optimizer state (optional)
        scheduler: Learning rate scheduler state (optional)
        epoch: Current epoch (optional)
        global_step: Current global step (optional)
        metadata: Additional metadata to save (optional)
        **kwargs: Additional arguments
        
    Returns:
        str: Path where checkpoint was saved
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create comprehensive payload
    payload = {
        'cfg': kwargs.get('cfg', None),
        'state_dicts': {
            'model': model.state_dict(),
        },
        'pickles': {
            'metadata': metadata or {},
            'epoch': epoch,
            'global_step': global_step,
        }
    }
    
    # Add LoRA-specific components if available
    if hasattr(model, 'lora_model'):
        payload['state_dicts']['lora_model'] = model.lora_model.state_dict()
    if hasattr(model, 'lora_encoder'):
        payload['state_dicts']['lora_encoder'] = model.lora_encoder.state_dict()
    
    # Add optimizer and scheduler if provided
    if optimizer is not None:
        payload['state_dicts']['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        payload['state_dicts']['scheduler'] = scheduler.state_dict()
    
    # Add LoRA configuration if available
    if hasattr(model, 'lora_model') and hasattr(model.lora_model, 'peft_config'):
        payload['pickles']['lora_config'] = {
            'lora_r': model.lora_model.peft_config['default'].r,
            'lora_alpha': model.lora_model.peft_config['default'].lora_alpha,
            'lora_dropout': model.lora_model.peft_config['default'].lora_dropout,
            'target_modules': model.lora_model.peft_config['default'].target_modules,
            'task_type': str(model.lora_model.peft_config['default'].task_type),
        }
    
    # Add model configuration if available
    if hasattr(model, 'task'):
        payload['pickles']['model_config'] = {
            'task': model.task,
            'cond_mechanism': getattr(model, 'cond_mechanism', None),
            'lora_encoder_units': getattr(model, 'lora_encoder_units', None),
            'apply_to': getattr(model, 'apply_to', None),
            'target_module_list': getattr(model, 'target_module_list', None),
        }
    
    torch.save(payload, path.open('wb'), pickle_module=dill)
    print(f"LoRA checkpoint saved to: {path}")
    return str(path.absolute())


def load_lora_checkpoint(
    path: Union[str, pathlib.Path],
    model: Optional[Any] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    load_optimizer: bool = True,
    load_scheduler: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Load a LoRA checkpoint with options.
    
    Args:
        path: Path to load the checkpoint from
        model: Model to load state into (optional)
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        load_optimizer: Whether to load optimizer state
        load_scheduler: Whether to load scheduler state
        **kwargs: Additional arguments
        
    Returns:
        Dict: Loaded checkpoint payload
    """
    path = pathlib.Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    
    payload = torch.load(path.open('rb'), pickle_module=dill, **kwargs)
    
    # Load model state if model provided
    if model is not None and 'state_dicts' in payload:
        if 'model' in payload['state_dicts']:
            model.load_state_dict(payload['state_dicts']['model'])
        
        # Load LoRA-specific components
        if hasattr(model, 'lora_model') and 'lora_model' in payload['state_dicts']:
            model.lora_model.load_state_dict(payload['state_dicts']['lora_model'])
        if hasattr(model, 'lora_encoder') and 'lora_encoder' in payload['state_dicts']:
            model.lora_encoder.load_state_dict(payload['state_dicts']['lora_encoder'])
    
    # Load optimizer state if provided and requested
    if optimizer is not None and load_optimizer and 'optimizer' in payload.get('state_dicts', {}):
        optimizer.load_state_dict(payload['state_dicts']['optimizer'])
    
    # Load scheduler state if provided and requested
    if scheduler is not None and load_scheduler and 'scheduler' in payload.get('state_dicts', {}):
        scheduler.load_state_dict(payload['state_dicts']['scheduler'])
    
    print(f"LoRA checkpoint loaded from: {path}")
    return payload


def create_lora_model_from_checkpoint(
    checkpoint_path: Union[str, pathlib.Path],
    base_model_class,
    base_model_kwargs: Dict[str, Any],
    lora_model_class=None,
    **kwargs
):
    """
    Create a LoRA model instance from a checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint
        base_model_class: Class for the base model
        base_model_kwargs: Arguments for base model creation
        lora_model_class: Class for the LoRA model (optional)
        **kwargs: Additional arguments
        
    Returns:
        LoRA model instance
    """
    path = pathlib.Path(checkpoint_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    
    payload = torch.load(path.open('rb'), pickle_module=dill)
    
    # Extract configuration from checkpoint
    model_config = payload.get('pickles', {}).get('model_config', {})
    lora_config = payload.get('pickles', {}).get('lora_config', {})
    
    # Merge configurations
    config = {**base_model_kwargs, **model_config, **kwargs}
    
    # Create base model
    base_model = base_model_class(**base_model_kwargs)
    
    # Create LoRA model
    if lora_model_class is None:
        # Try to import the LoRA model class
        from pdp.lora_model import LoraTransformerForDiffusion
        lora_model_class = LoraTransformerForDiffusion
    
    lora_model = lora_model_class(
        base_model=base_model,
        **config
    )
    
    # Load state dicts
    if 'state_dicts' in payload:
        if 'model' in payload['state_dicts']:
            lora_model.load_state_dict(payload['state_dicts']['model'])
        if 'lora_model' in payload['state_dicts']:
            lora_model.lora_model.load_state_dict(payload['state_dicts']['lora_model'])
        if 'lora_encoder' in payload['state_dicts']:
            lora_model.lora_encoder.load_state_dict(payload['state_dicts']['lora_encoder'])
    
    print(f"LoRA model created from checkpoint: {checkpoint_path}")
    return lora_model


def merge_lora_weights(lora_model, alpha: float = 1.0):
    """
    Merge LoRA weights into the base model.
    
    Args:
        lora_model: LoRA model to merge
        alpha: Scaling factor for LoRA weights
        
    Returns:
        Base model with merged weights
    """
    # Create a copy of the base model
    merged_model = lora_model.base_model
    
    # Get LoRA adapters
    lora_adapters = lora_model.lora_model
    
    # Merge LoRA weights into base model
    for name, module in merged_model.named_modules():
        if hasattr(module, 'weight') and hasattr(module, 'bias'):
            # Find corresponding LoRA adapter
            lora_name = name.replace('base_model.', '')
            if lora_name in lora_adapters.named_modules():
                # Apply LoRA scaling and merge
                lora_module = dict(lora_adapters.named_modules())[lora_name]
                if hasattr(lora_module, 'lora_A') and hasattr(lora_module, 'lora_B'):
                    # LoRA weight = lora_B @ lora_A * scaling
                    lora_weight = lora_module.lora_B @ lora_module.lora_A * alpha
                    module.weight.data += lora_weight
    
    print(f"LoRA weights merged into base model with alpha={alpha}")
    return merged_model


def save_merged_model(merged_model, path: Union[str, pathlib.Path]):
    """
    Save a merged model (base model with LoRA weights).
    
    Args:
        merged_model: Merged model to save
        path: Path to save the model
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    payload = {
        'state_dict': merged_model.state_dict(),
        'model_type': 'merged_lora',
    }
    
    torch.save(payload, path.open('wb'), pickle_module=dill)
    print(f"Merged model saved to: {path}")


def compare_lora_configs(config1: Dict[str, Any], config2: Dict[str, Any]) -> bool:
    """
    Compare two LoRA configurations for compatibility.
    
    Args:
        config1: First configuration
        config2: Second configuration
        
    Returns:
        bool: True if configurations are compatible
    """
    # Key parameters that must match
    critical_params = ['lora_r', 'lora_alpha', 'target_modules']
    
    for param in critical_params:
        if config1.get(param) != config2.get(param):
            return False
    
    return True


def get_lora_model_info(model) -> Dict[str, Any]:
    """
    Get information about a LoRA model.
    
    Args:
        model: LoRA model
        
    Returns:
        Dict: Model information
    """
    info = {
        'model_type': 'lora',
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'frozen_parameters': sum(p.numel() for p in model.parameters() if not p.requires_grad),
    }
    
    if hasattr(model, 'lora_model') and hasattr(model.lora_model, 'peft_config'):
        peft_config = model.lora_model.peft_config['default']
        info['lora_config'] = {
            'r': peft_config.r,
            'alpha': peft_config.lora_alpha,
            'dropout': peft_config.lora_dropout,
            'target_modules': peft_config.target_modules,
            'task_type': str(peft_config.task_type),
        }
    
    if hasattr(model, 'task'):
        info['task'] = model.task
    if hasattr(model, 'apply_to'):
        info['apply_to'] = model.apply_to
    if hasattr(model, 'target_module_list'):
        info['target_module_list'] = model.target_module_list
    
    return info


def print_lora_model_summary(model):
    """
    Print a summary of the LoRA model.
    
    Args:
        model: LoRA model to summarize
    """
    info = get_lora_model_info(model)
    
    print("LoRA Model Summary:")
    print("=" * 50)
    print(f"Model Type: {info['model_type']}")
    print(f"Total Parameters: {info['total_parameters']:,}")
    print(f"Trainable Parameters: {info['trainable_parameters']:,}")
    print(f"Frozen Parameters: {info['frozen_parameters']:,}")
    print(f"Trainable Ratio: {info['trainable_parameters'] / info['total_parameters']:.2%}")
    
    if 'lora_config' in info:
        print(f"\nLoRA Configuration:")
        for key, value in info['lora_config'].items():
            print(f"  {key}: {value}")
    
    if 'task' in info:
        print(f"\nTask: {info['task']}")
    if 'apply_to' in info:
        print(f"Apply To: {info['apply_to']}")
    if 'target_module_list' in info:
        print(f"Target Modules: {info['target_module_list']}")
    
    print("=" * 50)


