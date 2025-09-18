# LoRA Model Save/Load Guide

This guide explains how to save and load LoRA (Low-Rank Adaptation) models in the PDP framework.

## Overview

LoRA models in this framework are built on top of PEFT (Parameter-Efficient Fine-Tuning) and provide several ways to save and load model checkpoints:

1. **LoRA Adapters Only**: Save only the LoRA adapter weights (most efficient)
2. **Full Checkpoint**: Save complete model including base model and LoRA adapters
3. **Framework Integration**: Use with the existing workspace checkpoint system
4. **Utility Functions**: Helper functions for common operations

## Quick Start

### Basic Usage

```python
from pdp.lora_model import LoraTransformerForDiffusion
from pdp.modules_v2 import QKVTransformerForDiffusion

# Create base model
base_model = QKVTransformerForDiffusion(...)

# Create LoRA model
lora_model = LoraTransformerForDiffusion(
    base_model=base_model,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=['q_proj', 'k_proj', 'out_proj'],
    apply_to='both'
)

# Save LoRA adapters only (most efficient)
lora_model.save_lora_adapters('lora_adapters.pth')

# Save full checkpoint
lora_model.save_full_checkpoint('lora_full.pth')

# Load LoRA adapters
lora_model.load_lora_adapters('lora_adapters.pth')

# Load full checkpoint
lora_model.load_full_checkpoint('lora_full.pth')
```

## Detailed Methods

### 1. LoRA Adapters Only

**When to use**: When you want to save only the LoRA adapter weights for maximum efficiency.

```python
# Save only LoRA adapters
lora_model.save_lora_adapters('adapters.pth')

# Load LoRA adapters
lora_model.load_lora_adapters('adapters.pth')
```

**Advantages**:
- Smallest file size
- Fastest save/load
- Only contains trainable parameters

**Disadvantages**:
- Requires base model to be available separately
- Configuration must match exactly

### 2. Full Checkpoint

**When to use**: When you want to save everything in one file for easy distribution.

```python
# Save full checkpoint
lora_model.save_full_checkpoint('full_checkpoint.pth')

# Load full checkpoint
lora_model.load_full_checkpoint('full_checkpoint.pth')
```

**Advantages**:
- Self-contained (includes base model)
- Easy to share and distribute
- Compatible with standard PyTorch loading

**Disadvantages**:
- Larger file size
- Slower save/load

### 3. Class Method Loading

**When to use**: When you want to create a new LoRA model instance from a checkpoint.

```python
# Create LoRA model from checkpoint
lora_model = LoraTransformerForDiffusion.from_checkpoint(
    checkpoint_path='checkpoint.pth',
    base_model_path=None  # Optional: separate base model path
)
```

**Advantages**:
- One-step model creation and loading
- Handles configuration automatically
- Good for inference scenarios

### 4. Workspace Integration

**When to use**: When working with the PDP training framework.

```python
from pdp.workspace import DiffusionPolicyWorkspace

# Create workspace with LoRA model
workspace = DiffusionPolicyWorkspace(cfg)

# Check if model is LoRA
if workspace.is_lora_model():
    # Save LoRA checkpoint
    workspace.save_lora_checkpoint('workspace_lora.pth')
    
    # Load LoRA checkpoint
    workspace.load_lora_checkpoint('workspace_lora.pth')
else:
    # Use standard checkpoint methods
    workspace.save_checkpoint('standard.pth')
```

### 5. Utility Functions

**When to use**: For advanced checkpoint handling and batch operations.

```python
from pdp.utils.lora_utils import (
    save_lora_checkpoint,
    load_lora_checkpoint,
    create_lora_model_from_checkpoint,
    merge_lora_weights,
    save_merged_model
)

# Save with additional metadata
save_lora_checkpoint(
    model=lora_model,
    path='checkpoint.pth',
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=10,
    global_step=1000,
    metadata={'description': 'Fine-tuned model'}
)

# Load with specific options
payload = load_lora_checkpoint(
    path='checkpoint.pth',
    model=lora_model,
    optimizer=optimizer,
    load_optimizer=True,
    strict=False
)

# Merge LoRA weights into base model
merged_model = merge_lora_weights(lora_model, alpha=1.0)
save_merged_model(merged_model, 'merged_model.pth')
```

## Configuration Compatibility

When loading LoRA models, the framework checks for configuration compatibility:

```python
# Configuration mismatch example
lora_config = {
    'lora_r': 8,
    'lora_alpha': 16,
    'lora_dropout': 0.1,
    'target_modules': ['q_proj', 'k_proj', 'out_proj'],
    'apply_to': 'both'
}

# The framework will warn if configurations don't match
# but will still attempt to load compatible parameters
```

## Error Handling

The framework provides detailed error messages for common issues:

```python
try:
    lora_model.load_lora_adapters('adapters.pth')
except Exception as e:
    print(f"Error loading LoRA adapters: {e}")
    # Check for:
    # - Configuration mismatches
    # - Missing files
    # - Incompatible model architectures
```

## Best Practices

### 1. Choose the Right Save Method

- **Training**: Use `save_lora_checkpoint()` with workspace
- **Inference**: Use `save_lora_adapters()` for efficiency
- **Distribution**: Use `save_full_checkpoint()` for self-contained files

### 2. Configuration Management

```python
# Save configuration with checkpoint
lora_config = {
    'lora_r': 8,
    'lora_alpha': 16,
    'lora_dropout': 0.1,
    'target_modules': ['q_proj', 'k_proj', 'out_proj'],
    'apply_to': 'both'
}

# Always verify configuration when loading
if not compare_lora_configs(saved_config, current_config):
    print("Warning: Configuration mismatch!")
```

### 3. Memory Management

```python
# For large models, use CPU for checkpoint operations
lora_model.save_lora_adapters('adapters.pth')  # Automatically moves to CPU

# Load on specific device
lora_model.load_lora_adapters('adapters.pth')  # Loads on CPU first
lora_model.to('cuda')  # Move to GPU after loading
```

### 4. Version Control

```python
# Include version information in metadata
metadata = {
    'version': '1.0.0',
    'framework': 'PDP',
    'peft_version': '0.7.0',
    'description': 'Fine-tuned LoRA model'
}

save_lora_checkpoint(
    model=lora_model,
    path='checkpoint.pth',
    metadata=metadata
)
```

## Troubleshooting

### Common Issues

1. **Configuration Mismatch**
   ```
   Warning: LoRA configuration mismatch!
   ```
   - Check that LoRA parameters match between save and load
   - Verify target modules are the same

2. **Missing Keys**
   ```
   Missing keys: ['lora_A.weight', 'lora_B.weight']
   ```
   - Ensure LoRA adapters were properly saved
   - Check that model architecture matches

3. **Unexpected Keys**
   ```
   Unexpected keys: ['base_model.weight']
   ```
   - Normal when loading full checkpoint into adapter-only model
   - Can be safely ignored with `strict=False`

### Debug Tips

```python
# Print model information
lora_model.print_targeted_modules()

# Check trainable parameters
trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params}")

# Verify LoRA configuration
config = lora_model.lora_model.peft_config['default']
print(f"LoRA config: {config}")
```

## Examples

See `examples/lora_save_load_example.py` for comprehensive examples covering all use cases.

## API Reference

### LoraTransformerForDiffusion Methods

- `save_lora_adapters(path)`: Save only LoRA adapter weights
- `load_lora_adapters(path)`: Load LoRA adapter weights
- `save_full_checkpoint(path)`: Save complete model checkpoint
- `load_full_checkpoint(path)`: Load complete model checkpoint
- `from_checkpoint(checkpoint_path, base_model_path=None)`: Class method to create model from checkpoint

### Workspace Methods

- `is_lora_model()`: Check if model is LoRA
- `save_lora_checkpoint(path, tag, use_thread)`: Save LoRA checkpoint
- `load_lora_checkpoint(path, tag, load_optimizer)`: Load LoRA checkpoint

### Utility Functions

- `save_lora_checkpoint(model, path, ...)`: Save with metadata
- `load_lora_checkpoint(path, model, ...)`: Load with options
- `create_lora_model_from_checkpoint(...)`: Create model from checkpoint
- `merge_lora_weights(model, alpha)`: Merge LoRA into base model
- `save_merged_model(model, path)`: Save merged model
- `compare_lora_configs(config1, config2)`: Compare configurations
