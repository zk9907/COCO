import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter as NNParameter
from typing import Iterable, List, Optional


class LinearWithLoRA(nn.Linear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 r: int = 8,
                 alpha: int = 16,
                 dropout: float = 0.0,
                 freeze_base: bool = True):
        super().__init__(in_features, out_features, bias=bias)
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.scaling = alpha / float(r) if r > 0 else 0.0
        self.lora_dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.freeze_base = freeze_base

        # Initialize LoRA params: A ~ Kaiming, B = 0 so initial delta=0
        if r > 0:
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            with torch.no_grad():
                self.lora_B.weight.zero_()

        if freeze_base:
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = super().forward(input)
        if self.scaling == 0.0:
            return result
        lora_update = self.lora_B(self.lora_A(self.lora_dropout(input)))
        return result + self.scaling * lora_update


def _iter_named_modules_with_parents(module: nn.Module, prefix: str = ""):
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        yield module, name, child, full_name
        # recurse
        for p, n, c, f in _iter_named_modules_with_parents(child, full_name):
            yield p, n, c, f


def apply_lora(model: nn.Module,
               r: int = 8,
               alpha: int = 16,
               dropout: float = 0.05,
               target_modules: Optional[Iterable[str]] = None,
               freeze_base: bool = True) -> List[NNParameter]:
    """
    Replace selected nn.Linear layers with LinearWithLoRA, preserving existing weights.

    - target_modules: list of substrings; any linear layer whose qualified name contains
      one of these substrings will be converted. If None or empty, all Linear are converted.
    Returns the list of LoRA parameters for optimizer convenience.
    """
    if target_modules is not None:
        target_modules = list(target_modules)

    lora_params: List[NNParameter] = []
    to_convert: List[tuple] = []

    for parent, child_name, child, full_name in _iter_named_modules_with_parents(model):
        if not isinstance(child, nn.Linear):
            continue
        if target_modules and not any(t in full_name for t in target_modules):
            continue
        to_convert.append((parent, child_name, child))

    for parent, child_name, linear in to_convert:
        new_linear = LinearWithLoRA(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            r=r,
            alpha=alpha,
            dropout=dropout,
            freeze_base=freeze_base,
        )
        # Ensure new module is on the same device as the original linear
        new_linear = new_linear.to(linear.weight.device)
        # copy base weights/bias
        with torch.no_grad():
            new_linear.weight.copy_(linear.weight.data)
            if linear.bias is not None and new_linear.bias is not None:
                new_linear.bias.copy_(linear.bias.data)
        setattr(parent, child_name, new_linear)
        lora_params.extend(list(new_linear.lora_A.parameters()))
        lora_params.extend(list(new_linear.lora_B.parameters()))

    return lora_params


def mark_only_lora_as_trainable(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, LinearWithLoRA):
            for p in module.lora_A.parameters():
                p.requires_grad = True
            for p in module.lora_B.parameters():
                p.requires_grad = True


def get_lora_parameters(model: nn.Module) -> List[NNParameter]:
    params: List[NNParameter] = []
    for module in model.modules():
        if isinstance(module, LinearWithLoRA):
            params.extend(list(module.lora_A.parameters()))
            params.extend(list(module.lora_B.parameters()))
    return params


def count_lora_parameters(model):
    """Count total and LoRA parameters in a model"""
    total_params = sum(p.numel() for p in model.parameters())
    lora_params = sum(p.numel() for p in get_lora_parameters(model))
    return total_params, lora_params