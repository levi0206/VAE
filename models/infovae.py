import torch
from torch import nn
import torch.nn.functional as F
from typing import List
from lib.utils import sample_indices

class InfoVAE(nn.Module):
    def __init__(self, x_aug_sig, epoch, batch_size, device, hidden_dims: List, kernel_width=1.0, lambda_=10, alpha=0.5):
        super(InfoVAE, self).__init__()

        print("Input tensor shape: {}".format(x_aug_sig.shape))
        print("Hidden dims: {}".format(hidden_dims))

        self.x_aug_sig = x_aug_sig
        self.epoch = epoch
        self.batch_size = batch_size
        self.device = device
        self.hidden_dims = hidden_dims
        self.kernel_width = kernel_width
        self.lambda_ = lambda_  # Weight for prior matching term
        self.alpha = alpha      # Weight for mutual information term

        # Encoder & Decoder
        self.encoder_mu = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LeakyReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.LeakyReLU(),
        )
        self.encoder_sigma = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LeakyReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims[2], hidden_dims[1]),
            nn.LeakyReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.LeakyReLU(),
        )

        # Move to device
        self.encoder_mu.to(device)
        self.encoder_sigma.to(device)
        self.decoder.to(device)

    def encode(self, x):
        x_flatten = x.view(x.shape[0], -1)
        mean = self.encoder_mu(x_flatten)
        log_var = self.encoder_sigma(x_flatten)
        log_var = torch.clamp(log_var, min=-10, max=10)
        noise = torch.randn_like(mean).to(self.device)
        z = mean + torch.exp(0.5 * log_var) * noise
        return mean, log_var, z
        
    def decode(self, z):
        return self.decoder(z)

    def gaussian_kernel(self, x, y):
        """Computes the Gaussian kernel between two distributions."""
        xx = torch.sum(x**2, 1).view(-1, 1)
        yy = torch.sum(y**2, 1).view(1, -1)
        dist = xx + yy - 2 * torch.mm(x, y.t())
        return torch.exp(-dist / (2 * self.kernel_width**2))

    def compute_mmd(self, z, z_prior):
        """Computes the Maximum Mean Discrepancy (MMD)."""
        K_zz = self.gaussian_kernel(z, z)
        K_zp_zp = self.gaussian_kernel(z_prior, z_prior)
        K_z_zp = self.gaussian_kernel(z, z_prior)
        return K_zz.mean() + K_zp_zp.mean() - 2 * K_z_zp.mean()

    def loss(self, mean, log_var, sample_data, reconstructed_data, z, z_prior):
        # Reconstruction loss
        recon_loss = F.mse_loss(sample_data, reconstructed_data, reduction='mean')

        # Mutual Information Term: KL divergence between q(z|x) and q(z)
        mutual_info_loss = 0.5 * (log_var.exp() + mean.pow(2) - 1 - log_var).sum(dim=1).mean()

        # Prior matching term (MMD)
        prior_matching_loss = self.compute_mmd(z, z_prior)

        # InfoVAE loss
        loss = recon_loss + self.lambda_ * prior_matching_loss + self.alpha * mutual_info_loss
        return loss

    def generate(self, x: torch.Tensor):
        _, _, z = self.encode(x)
        return self.decode(z)

def InfoVAE_train(model, optimizer):
    early_stop = 400
    cnt = 0
    min_loss = float('inf')

    for i in range(model.epoch):
        # Sample batch
        time_indics = sample_indices(model.x_aug_sig.shape[0], model.batch_size, "cuda")
        sample_data = model.x_aug_sig[time_indics]
        
        # Sample prior (Gaussian)
        z_prior = torch.randn(model.batch_size, model.hidden_dims[2]).to(model.device)

        # Forward pass
        mean, log_var, z = model.encode(sample_data)
        reconstructed_data = model.decode(z)
        
        # Compute loss
        loss = model.loss(mean, log_var, sample_data.view(model.batch_size, -1), reconstructed_data, z, z_prior)

        # Backpropagation
        optimizer.zero_grad()
        if loss < min_loss:
            loss.backward()
            optimizer.step()

        # Print loss
        if i % 100 == 0:
            print(f"Epoch {i} loss {loss.item():.4f}")

        # Early stopping
        if loss.item() < min_loss:
            min_loss = loss.item()
            cnt = 0
        else:
            cnt += 1
            if cnt > early_stop:
                print(f"min_loss: {min_loss:.4f}")
                break
