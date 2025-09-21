import torch
import torch.nn as nn
from pdp.modules import QKVTransformerForDiffusion


from peft import get_peft_model, LoraConfig, TaskType


class LoraTransformerForDiffusion(nn.Module):
    """
    Wraps a QKVTransformerForDiffusion model with PEFT LoRA adapters.
    """
    def __init__(
        self,
        base_model: QKVTransformerForDiffusion,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target_modules: list = ['q_proj', 'k_proj', 'out_proj'],
        apply_to: str = "both",  # "encoder", "decoder", or "both"
        **kwargs
    ):
        super().__init__()
        if get_peft_model is None:
            raise ImportError("peft is not installed. Please install peft to use LoraTransformerForDiffusion.")

        self.base_model = base_model
        # Disable gradients for all parameters in the base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.apply_to = apply_to

        # Validate apply_to parameter
        if apply_to not in ["encoder", "decoder", "both"]:
            raise ValueError("apply_to must be one of: 'encoder', 'decoder', 'both'")

        # Default target_modules: all Linear layers in the model
        target_modules = self._find_linear_module_names(self.base_model, apply_to, target_modules)

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            task_type=TaskType.SEQ_2_SEQ_LM  # Better suited for encoder-decoder architecture
        )

        self.lora_model = get_peft_model(self.base_model, lora_config)

    def forward(self, *args, **kwargs):
        return self.lora_model(*args, **kwargs)

    def _find_linear_module_names(self, model, apply_to="both", target_modules=[]):
        """
        Recursively find nn.Linear module names based on which part of the model to target.
        
        Args:
            model: The QKVTransformerForDiffusion model
            apply_to: "encoder", "decoder", or "both"
        """
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
                    for target in target_modules:
                        if target in name:
                            linear_names.add(name)
                  
        print('Implimenting LoRA on the following modules:')
        for name in linear_names:
            print(f"  - {name}")
        print('===================================')
        return list(linear_names)

    def get_targeted_modules(self):
        """
        Return the list of modules that have LoRA adapters applied.
        Useful for debugging and understanding what's being fine-tuned.
        """
        return self._find_linear_module_names(self.base_model, self.apply_to)

    def print_targeted_modules(self):
        """
        Print which modules are being targeted by LoRA adapters.
        """
        modules = self.get_targeted_modules()
        print(f"LoRA adapters applied to {self.apply_to}:")
        for module in sorted(modules):
            print(f"  - {module}")
        print(f"Total modules: {len(modules)}")

    def get_optim_groups(self, weight_decay):
        """
        Return optimizer groups for LoRA parameters only.
        """
        decay = []
        no_decay = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "layernorm" in name:
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def set_normalizer(self, normalizer):
        if hasattr(self.base_model, "set_normalizer"):
            self.base_model.set_normalizer(normalizer)

    def save_lora_adapters(self, path):
        """
        Save only the LoRA adapter weights to a file.
        This is more efficient than saving the entire model.
        
        Args:
            path (str): Path to save the LoRA adapters
        """
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Get only the LoRA adapter parameters
        lora_state_dict = {}
        for name, param in self.lora_model.named_parameters():
            if param.requires_grad:  # Only LoRA parameters have requires_grad=True
                lora_state_dict[name] = param.detach().cpu()
        
        # Also save LoRA configuration for loading
        lora_config = {
            'lora_r': self.lora_model.peft_config['default'].r,
            'lora_alpha': self.lora_model.peft_config['default'].lora_alpha,
            'lora_dropout': self.lora_model.peft_config['default'].lora_dropout,
            'target_modules': self.lora_model.peft_config['default'].target_modules,
            'apply_to': self.apply_to,
            'lora_state_dict': lora_state_dict
        }
        
        torch.save(lora_config, path)
        print(f"LoRA adapters saved to: {path}")

    def load_lora_adapters(self, path):
        """
        Load LoRA adapter weights from a file.
        
        Args:
            path (str): Path to load the LoRA adapters from
        """
        lora_config = torch.load(path, map_location='cpu')
        
        # Verify configuration matches
        current_config = self.lora_model.peft_config['default']
        if (lora_config['lora_r'] != current_config.r or 
            lora_config['lora_alpha'] != current_config.lora_alpha or
            lora_config['lora_dropout'] != current_config.lora_dropout or
            lora_config['target_modules'] != current_config.target_modules):
            print("Warning: LoRA configuration mismatch!")
            print(f"Saved config: {lora_config}")
            print(f"Current config: {current_config}")
        
        # Load the LoRA state dict
        lora_state_dict = lora_config['lora_state_dict']
        
        # Load only the LoRA parameters
        missing_keys, unexpected_keys = self.lora_model.load_state_dict(lora_state_dict)
        
        if missing_keys:
            print(f"Missing LoRA keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected LoRA keys: {unexpected_keys}")
        
        print(f"LoRA adapters loaded from: {path}")

    def save_full_checkpoint(self, path):
        """
        Save the complete model checkpoint including base model and LoRA adapters.
        This is compatible with the framework's standard checkpoint system.
        
        Args:
            path (str): Path to save the full checkpoint
        """
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the full model state dict (includes both base and LoRA parameters)
        checkpoint = {
            'model_state_dict': self.lora_model.state_dict(),
            'lora_config': {
                'lora_r': self.lora_model.peft_config['default'].r,
                'lora_alpha': self.lora_model.peft_config['default'].lora_alpha,
                'lora_dropout': self.lora_model.peft_config['default'].lora_dropout,
                'target_modules': self.lora_model.peft_config['default'].target_modules,
                'apply_to': self.apply_to,
            }
        }
        
        torch.save(checkpoint, path)
        print(f"Full LoRA checkpoint saved to: {path}")

    def load_full_checkpoint(self, path):
        """
        Load a complete model checkpoint including base model and LoRA adapters.
        
        Args:
            path (str): Path to load the full checkpoint from
        """
        checkpoint = torch.load(path, map_location='cpu')
        
        # Load the full model state dict
        missing_keys, unexpected_keys = self.lora_model.load_state_dict(
            checkpoint['model_state_dict'], strict=False
        )
        
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        
        print(f"Full LoRA checkpoint loaded from: {path}")

    @classmethod
    def from_checkpoint(cls, checkpoint_path, base_model_path=None, **kwargs):
        """
        Create a LoraTransformerForDiffusion instance from a saved checkpoint.
        
        Args:
            checkpoint_path (str): Path to the LoRA checkpoint
            base_model_path (str, optional): Path to base model if different from checkpoint
            **kwargs: Additional arguments for LoRA configuration
            
        Returns:
            LoraTransformerForDiffusion: Loaded model instance
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        lora_config = checkpoint['lora_config']
        
        # Load base model if path provided
        if base_model_path:
            base_model = torch.load(base_model_path, map_location='cpu')
            if isinstance(base_model, dict):
                base_model = base_model['model_state_dict']
        else:
            # Assume base model is in the same checkpoint
            base_model = checkpoint['model_state_dict']
        
        # Create LoRA model with saved configuration
        instance = cls(
            base_model=base_model,
            lora_r=lora_config['lora_r'],
            lora_alpha=lora_config['lora_alpha'],
            lora_dropout=lora_config['lora_dropout'],
            target_modules=lora_config['target_modules'],
            apply_to=lora_config['apply_to'],
            **kwargs
        )
        
        # Load the LoRA weights
        instance.load_full_checkpoint(checkpoint_path)
        
        return instance

