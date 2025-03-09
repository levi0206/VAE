# Source: 
# https://github.com/luchungi/Generative-Model-Signature-MMD/blob/main/sigkernel/kernels.py
# https://github.com/luchungi/Generative-Model-Signature-MMD/blob/main/sigkernel/loss.py

import torch
from typing import Optional
from abc import ABCMeta, abstractmethod
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

class LinearKernel(Kernel):

    def __init__(self):
        super().__init__()
        self.static_kernel_type = 'linear'

    def gram_matrix(self, X: torch.Tensor, Y: torch.Tensor = None) -> torch.Tensor:
        return matrix_mult(X, Y, transpose_Y=True)
    

def matrix_mult(X : torch.Tensor, Y : Optional[torch.Tensor] = None, transpose_X : bool = False, transpose_Y : bool = False) -> torch.Tensor:
    subscript_X = '...ji' if transpose_X else '...ij'
    subscript_Y = '...kj' if transpose_Y else '...jk'
    return torch.einsum(f'{subscript_X},{subscript_Y}->...ik', X, Y if Y is not None else X)

class SignatureKernel():
    def __init__(self, n_levels: int = 5, static_kernel: Optional[Kernel] = None) -> None:
        '''
        Parameters
        ----------
        n_levels: int, default=4
            The number of levels of the signature to keep. Higher order terms are truncated
        static_kernel: Kernel, default=None
            The kernel to use for the static kernel. If None, the linear kernel is used.
        
        Comments
        ----------
        If the input tensor is a signature, reshape it to [batch_size,-1]
        '''

        self.n_levels = n_levels
        self.static_kernel = static_kernel if static_kernel is not None else LinearKernel()

    def __call__(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        # print(X.reshape((-1, X.shape[-1])).shape)
        M = self.static_kernel(X.reshape((-1, X.shape[-1])), Y.reshape((-1, Y.shape[-1]))).reshape((X.shape[0], X.shape[1], Y.shape[0], Y.shape[1]))
        M = torch.diff(torch.diff(M, dim=1), dim=-1) # M[i,j,k,l] = k(X[i,j+1], Y[k,l+1]) - k(X[i,j], Y[k,l+1]) - k(X[i,j+1], Y[k,l]) + k(X[i,j], Y[k,l])
        n_X, n_Y = M.shape[0], M.shape[2]
        K = torch.ones((n_X, n_Y), dtype=M.dtype, device=M.device)
        K += torch.sum(M, dim=(1, -1))
        R = torch.clone(M)
        for _ in range(1, self.n_levels):
            R = M * multi_cumsum(R, axis=(1, -1))
            K += torch.sum(R, dim=(1, -1))

        return K

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

def mmd_loss(X: torch.tensor, Y: torch.tensor, kernel: SignatureKernel) -> torch.tensor:
    '''
    X: torch.tensor of shape (n_samples, n_features)
    Y: torch.tensor of shape (n_samples, n_features)
    kernel: kernel to be used e.g. SignatureKernel
    '''

    # calculate Gram matrices with normalisation and diagonal of XX/YY zeroed
    K_XX = kernel(X,X)
    K_YY = kernel(Y,Y)
    K_XY = kernel(X,Y)

    # unbiased MMD statistic (could also use biased, doesn't matter if we use permutation tests)
    n = len(K_XX)
    m = len(K_YY)

    mmd = (torch.sum(K_XX[~torch.eye(*K_XX.shape,dtype=torch.bool)]) / (n*(n-1))
           + torch.sum(K_YY[~torch.eye(*K_YY.shape, dtype=torch.bool)]) / (m*(m-1))
           - 2*torch.sum(K_XY)/(n*m))

    return mmd