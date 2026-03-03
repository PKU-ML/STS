import torch
import contextlib
import functools

from typing import List, Tuple, Callable
from torch import Tensor
import os

from sae_lens import SAE
activation_set = []


def calculate_minus_temp(features,sft_features):
    minus_eng = features - sft_features
    minus_eng = minus_eng**2
    minus_eng_sum = torch.sum(minus_eng,dim=0)
    return minus_eng_sum


@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs
):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_forward_pre_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward pre hook on the module
    module_forward_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()


def get_intervention_hook(
    sae: SAE,
    feature_idx: int,
    max_activation: float = 1.0,
    strength: float = 1.0,
):
    def hook_fn(module, input, output):
        if torch.is_tensor(output):
            activations = output.clone()
        else:
            activations = output[0].clone()

        if sae.device != activations.device:
            sae.device = activations.device
            sae.to(sae.device)

        

        features = sae.encode(activations)
        decoded_features =sae.decode(features)
        error = activations-decoded_features
        global activation_set
        if len(activation_set) == 0:
            activation_set = features.clone()
        print(activation_set.size())
        if len(activation_set)<40000:
            activation_set=torch.cat((activation_set,features))
            print(activation_set.size())
        else:
            torch.save(activation_set,'activation_qwen_limo_raw.pt')
            exit()
        
        print(activation_set.size())
        return output

    return hook_fn


