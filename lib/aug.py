# Source: https://github.com/SigCGANs/Conditional-Sig-Wasserstein-GANs/blob/master/lib/augmentations.py

import torch
from dataclasses import dataclass
from typing import List, Tuple

def lead_lag_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Lead-lag transformation. 
    See "A Primer on the Signature Method in Machine Learning"
    """
    # 0-th dim is batch
    repeat = torch.repeat_interleave(x, repeats=2, dim=1) 
    lead_lag = torch.cat([repeat[:, :-1], repeat[:, 1:]], dim=2)
    return lead_lag

def sig_normal(sig: torch.Tensor, normalize=False):
    if normalize == False:
        return sig.mean(0)
    elif normalize == True:
        mu = sig.mean(0)
        sigma = sig.std(0)
        sig = (sig-mu)/sigma
        return sig

@dataclass
class BaseAugmentation:
    pass

    def apply(self, *args: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError('Needs to be implemented by child.')
    
@dataclass
class LeadLag(BaseAugmentation):
    with_time: bool = False

    def apply(self, x: torch.Tensor):
        return lead_lag_transform(x)
        
def apply_augmentations(x: torch.Tensor, augmentations: Tuple) -> torch.Tensor:
    y = x.clone()
    for augmentation in augmentations:
        y = augmentation.apply(y)
        print(y.shape)
    return y

AUGMENTATIONS = {'LeadLag': LeadLag} 

def parse_augmentations(list_of_dicts):
    augmentations = list()
    for kwargs in list_of_dicts:
        name = kwargs.pop('name')
        augmentations.append(
            AUGMENTATIONS[name](**kwargs)
        )
    return augmentations