# Source: 
# https://github.com/luchungi/Generative-Model-Signature-MMD/blob/main/sigkernel/kernels.py
# https://github.com/luchungi/Generative-Model-Signature-MMD/blob/main/sigkernel/loss.py

import torch
import numpy as np
import torch

from abc import ABCMeta, abstractmethod
from typing import Optional

import numpy as np
import torch

class Kernel(metaclass=ABCMeta):
    '''
    Base class for static kernels.
    '''

    @abstractmethod
    def gram_matrix(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        pass

    def __call__(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return self.gram_matrix(X, Y)
    
class LinearKernel:

    def __call__(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return X @ Y.T

class RBFKernel(Kernel):

    def __init__(self, sigma: float = 1.0) -> None:
        super().__init__()
        self.sigma = sigma
        self.static_kernel_type = 'rbf'

    def gram_matrix(self, X: torch.Tensor, Y: torch.Tensor = None) -> torch.Tensor:
        D2_scaled = squared_euclid_dist(X, Y) / self.sigma**2
        return torch.exp(-D2_scaled)
    
def squared_norm(X : torch.Tensor, dim : int = -1) -> torch.Tensor:
    return torch.sum(torch.square(X), dim=dim)

def squared_euclid_dist(X : torch.Tensor, Y : Optional[torch.Tensor] = None) -> torch.Tensor:
    X_n2 = squared_norm(X)
    if Y is None:
        D2 = (X_n2[..., :, None] + X_n2[..., None, :]) - 2 * matrix_mult(X, X, transpose_Y=True)
    else:
        Y_n2 = squared_norm(Y, dim=-1)
        D2 = (X_n2[..., :, None] + Y_n2[..., None, :]) - 2 * matrix_mult(X, Y, transpose_Y=True)
    return D2

class SignatureKernel():
    def __init__(self, static_kernel: Optional[Kernel] = None) -> None:
        '''
        Parameters
        ----------
        static_kernel: Kernel, default=None
            The kernel to use for the signature-based kernel. If None, the linear kernel is used.
        '''
        self.static_kernel = static_kernel if static_kernel is not None else LinearKernel()

    def __call__(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        '''
        Computes the kernel matrix between signature features X and Y.

        X: torch.Tensor of shape (n_samples_X, feature_dim)
        Y: torch.Tensor of shape (n_samples_Y, feature_dim)

        Returns:
        A kernel matrix of shape (n_samples_X, n_samples_Y)
        '''
        return self.static_kernel(X, Y)  # Directly apply the static kernel

def multi_cumsum(M: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """Computes the exclusive cumulative sum along a given set of axes.

    Args:
        K (torch.Tensor): A matrix over which to compute the cumulative sum
        axis (int or iterable, optional): An axis or a collection of them. Defaults to -1 (the last axis).
    """

    ndim = M.ndim
    axis = [axis] if np.isscalar(axis) else axis
    axis = [ndim+ax if ax < 0 else ax for ax in axis]

    # create slice for exclusive cumsum (slice off last element along given axis then pre-pad with zeros)
    slices = tuple(slice(-1) if ax in axis else slice(None) for ax in range(ndim))
    M = M[slices]

    for ax in axis:
        M = torch.cumsum(M, dim=ax)

    # pre-pad with zeros along the given axis if exclusive cumsum
    pads = tuple(x for ax in reversed(range(ndim)) for x in ((1, 0) if ax in axis else (0, 0)))
    M = torch.nn.functional.pad(M, pads)

    return M

def mmd_loss(SigX: torch.Tensor, SigY: torch.Tensor, kernel: SignatureKernel) -> torch.tensor:
    '''
    Compute the MMD loss between two distributions using their signature representations.

    Parameters
    ----------
    SigX: torch.Tensor of shape (batch_size, signature_dim)
        Precomputed signatures for the first set of sequences.
    SigY: torch.Tensor of shape (batch_size, signature_dim)
        Precomputed signatures for the second set of sequences.
    kernel: SignatureKernel
        A kernel function operating on signatures.

    Returns
    -------
    torch.Tensor (scalar)
        The MMD loss between the two distributions.
    '''

    # Compute Gram matrices
    K_XX = kernel(SigX, SigX)
    K_YY = kernel(SigY, SigY)
    K_XY = kernel(SigX, SigY)

    # Unbiased MMD statistic (ensuring diagonal entries are not used)
    n = len(K_XX)
    m = len(K_YY)

    mmd = (torch.sum(K_XX[~torch.eye(n, dtype=torch.bool)]) / (n * (n - 1))
         + torch.sum(K_YY[~torch.eye(m, dtype=torch.bool)]) / (m * (m - 1))
         - 2 * torch.sum(K_XY) / (n * m))

    return max(mmd,0)

def matrix_mult(X : torch.Tensor, Y : Optional[torch.Tensor] = None, transpose_X : bool = False, transpose_Y : bool = False) -> torch.Tensor:
    subscript_X = '...ji' if transpose_X else '...ij'
    subscript_Y = '...kj' if transpose_Y else '...jk'
    return torch.einsum(f'{subscript_X},{subscript_Y}->...ik', X, Y if Y is not None else X)

# We implement the KL divergence, but we use nn.KLDivLoss(reduction="batchmean") in our testing phase.
def kl_divergence(P, Q, eps=1e-7):
    """
    Compute KL divergence between two sets of distributions.
    
    Args:
        P (torch.Tensor): Tensor of shape [batch, dim], the first distribution.
        Q (torch.Tensor): Tensor of shape [batch, dim], the second distribution.
        eps (float): Small constant to avoid numerical instability (default: 1e-10).
    
    Returns:
        kl (torch.Tensor): KL divergence for each batch, shape [batch].
        kl_scalar (torch.Tensor): Scalar KL divergence (mean over batches).
    """
    # Add epsilon to avoid log(0) or division by zero
    P = P + eps
    Q = Q + eps

    # Normalize over the dim dimension to ensure they are probability distributions
    P = P / P.sum(dim=-1, keepdim=True)
    Q = Q / Q.sum(dim=-1, keepdim=True)
    
    # Compute KL divergence: P * log(P / Q)
    kl = P * torch.log(P / Q)  # Element-wise operation
    print("kl: {}".format(kl))
    kl = kl.sum(dim=-1)        # Sum over dim, resulting in shape [batch]
    
    # Reduce to a scalar (mean over batches)
    kl_scalar = kl.mean()
    
    return kl, kl_scalar