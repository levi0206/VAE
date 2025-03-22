# Source: 
# https://github.com/luchungi/Generative-Model-Signature-MMD/blob/main/sigkernel/kernels.py
# https://github.com/luchungi/Generative-Model-Signature-MMD/blob/main/sigkernel/loss.py

import torch
import numpy as np
import torch

def mmd_loss(S_X: torch.Tensor, S_Y: torch.Tensor) -> torch.Tensor:
    '''
    S_X: torch.Tensor of shape (batch_X, dim) - precomputed signatures of paths
    S_Y: torch.Tensor of shape (batch_Y, dim) - precomputed signatures of paths
    '''
    # Compute Gram matrices using inner product of signatures
    K_XX = S_X @ S_X.T
    K_YY = S_Y @ S_Y.T
    K_XY = S_X @ S_Y.T

    # Get the number of samples in X and Y
    n = S_X.shape[0]
    m = S_Y.shape[0]

    # Compute the unbiased MMD statistic
    # Sum off-diagonal elements of K_XX and K_YY
    sum_XX = torch.sum(K_XX) - torch.trace(K_XX)
    sum_YY = torch.sum(K_YY) - torch.trace(K_YY)
    sum_XY = torch.sum(K_XY)

    # Compute MMD^2
    mmd_squared = (sum_XX / (n * (n - 1)) + sum_YY / (m * (m - 1)) - 2 * sum_XY / (n * m))

    return mmd_squared