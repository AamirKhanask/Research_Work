# hamer/callbacks/weight_logger.py
import torch
from pytorch_lightning.callbacks import Callback
import numpy as np
import os
import json


class BitLinearWeightLogger(Callback):
    """
    Weight logger that shows both FP32 stored weights AND ternary weights used in forward pass.
    weight_quant() is called ONLY for logging, NOT during training.
    """
    
    def __init__(self, log_subset_size=100, log_every_n_steps=None, verbose=False):
        super().__init__()
        self.log_subset_size = log_subset_size
        self.log_every_n_steps = log_every_n_steps
        self.verbose = verbose  # ? Add this flag        
        self.has_logged_initial = False
        
        # Import weight_quant only once (lazy import to avoid circular deps)
        self._weight_quant = None
    
    def _get_weight_quant(self):
        if self._weight_quant is None:
            from hamer.models.tokenmixers._vendor_mmfree.ops.fusedbitnet import weight_quant
            self._weight_quant = weight_quant
        return self._weight_quant
    

    def _get_bitlinear_weights(self, model):
        weight_quant_fn = self._get_weight_quant()
        weights_info = {}
        
        for name, module in model.named_modules():
            module_class = module.__class__.__name__
            if module_class in ['BitLinear', 'FusedBitLinear']:
                if hasattr(module, 'weight') and module.weight is not None:
                    fp32_weight = module.weight.data.clone().cpu().flatten()
                    
                    # Get the raw ternary values (-1, 0, +1) BEFORE dequantization
                    scale = 1.0 / module.weight.data.abs().mean().clamp_(min=1e-5)
                    raw_ternary = (module.weight.data * scale).round().clamp_(-1, 1)
                    raw_ternary = raw_ternary.cpu().flatten()  # Now contains -1, 0, or 1
                    
                    # Also get the dequantized version (what's actually used in computation)
                    dequantized_ternary = weight_quant_fn(module.weight.data).cpu().flatten()
                    
                    # Take random subset
                    if len(fp32_weight) > self.log_subset_size:
                        indices = torch.randperm(len(fp32_weight))[:self.log_subset_size]
                        fp32_subset = fp32_weight[indices].numpy()
                        raw_subset = raw_ternary[indices].numpy()
                        dequantized_subset = dequantized_ternary[indices].numpy()
                    else:
                        fp32_subset = fp32_weight.numpy()
                        raw_subset = raw_ternary.numpy()
                        dequantized_subset = dequantized_ternary.numpy()
                    
                    # Count raw ternary distribution (-1, 0, +1)
                    ternary_counts = {
                        '-1': int((raw_ternary == -1).sum().item()),
                        '0': int((raw_ternary == 0).sum().item()),
                        '+1': int((raw_ternary == 1).sum().item()),
                    }
                    
                    weights_info[name] = {
                        'shape': list(module.weight.shape),
                        'total_params': len(fp32_weight),
                        'fp32': {
                            'subset': fp32_subset.tolist(),
                            'mean': float(fp32_weight.mean().item()),
                            'std': float(fp32_weight.std().item()),
                            'min': float(fp32_weight.min().item()),
                            'max': float(fp32_weight.max().item()),
                        },
                        'ternary_raw': {  # This is what you want to see
                            'subset': raw_subset.tolist(),
                            'count_neg1': ternary_counts['-1'],
                            'count_zero': ternary_counts['0'],
                            'count_pos1': ternary_counts['+1'],
                            'percent_neg1': 100 * ternary_counts['-1'] / len(fp32_weight),
                            'percent_zero': 100 * ternary_counts['0'] / len(fp32_weight),
                            'percent_pos1': 100 * ternary_counts['+1'] / len(fp32_weight),
                        },
                        'ternary_dequantized': {  # For reference
                            'subset': dequantized_subset.tolist(),
                        }
                    }
        
        return weights_info
    
    def _save_weights(self, weights_info, stage, trainer):
        """Save weights to file."""
        if not weights_info:
            return
        
        output_dir = trainer.logger.log_dir if trainer.logger else '.'
        if not output_dir:
            output_dir = '.'
        
        log_file = os.path.join(output_dir, f'bitlinear_weights_{stage}.json')
        with open(log_file, 'w') as f:
            json.dump(weights_info, f, indent=2)
        
        # Print summary
        if self.verbose:        
          print(f"\n{'='*70}")
          print(f"BITLINEAR WEIGHTS - {stage.upper()}")
          print(f"Saved to: {log_file}")
          print(f"{'='*70}")
          
          for name, info in weights_info.items():
              print(f"\nLayer: {name}")
              print(f"  Shape: {info['shape']}")
              print(f"  Total params: {info['total_params']:,}")
              print(f"\n  FP32 (stored in memory):")
              print(f"    Mean: {info['fp32']['mean']:.6f}")
              print(f"    Std:  {info['fp32']['std']:.6f}")
              print(f"    Min:  {info['fp32']['min']:.6f}")
              print(f"    Max:  {info['fp32']['max']:.6f}")
              print(f"\n  Ternary (used in forward computation):")
              print(f"    -1: {info['ternary']['count_neg1']:,} ({info['ternary']['percent_neg1']:.2f}%)")
              print(f"     0: {info['ternary']['count_zero']:,} ({info['ternary']['percent_zero']:.2f}%)")
              print(f"    +1: {info['ternary']['count_pos1']:,} ({info['ternary']['percent_pos1']:.2f}%)")
              print(f"\n  Sample FP32 values (first 10): {info['fp32']['subset'][:10]}")
              print(f"  Sample Ternary values (first 10): {info['ternary']['subset'][:10]}")
          
          print(f"{'='*70}\n")
    
    def on_fit_start(self, trainer, pl_module):
        """Log weights at start of training."""
        if not self.has_logged_initial:
            weights_info = self._get_bitlinear_weights(pl_module)
            self._save_weights(weights_info, 'initial', trainer)
            self.has_logged_initial = True
    
    def on_fit_end(self, trainer, pl_module):
        """Log weights at end of training."""
        weights_info = self._get_bitlinear_weights(pl_module)
        self._save_weights(weights_info, 'final', trainer)