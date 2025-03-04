import torch
from torch import nn
import torch.nn.functional as F
from typing import List
from lib.utils import sample_indices,compute_mmd

class InfoVAE(nn.Module):
    def __init__(self, x_aug_sig, epoch, batch_size, hidden_dims: List, device) -> None:
        super(InfoVAE, self).__init__()

        self.x_aug_sig = x_aug_sig
        print("Input tensor shape: {}".format(x_aug_sig.shape))
        self.epoch = epoch
        self.batch_size = batch_size
        self.device = device

        # Assume len(hidden_dims)=3.
        self.encoder_mu = nn.Sequential(
            nn.Linear(hidden_dims[0],hidden_dims[1]),
            nn.LeakyReLU(),
            nn.Linear(hidden_dims[1],hidden_dims[2]),
            nn.LeakyReLU(),
        )
        self.encoder_sigma = nn.Sequential(
            nn.Linear(hidden_dims[0],hidden_dims[1]),
            nn.Tanh(),
            nn.Linear(hidden_dims[1],hidden_dims[2]),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims[2],hidden_dims[1]),
            nn.LeakyReLU(),
            nn.Linear(hidden_dims[1],hidden_dims[0]),
            nn.LeakyReLU(),
        )

        # To device
        self.encoder_mu.to(device)
        self.encoder_sigma.to(device)
        self.decoder.to(device)
    
    def encode(self, x):
        x_flatten = x.view(x.shape[0],-1)
        mean = self.encoder_mu(x_flatten)
        log_var = self.encoder_sigma(x_flatten)
        # Clipping
        log_var = torch.clamp(log_var, min=-10, max=10)
        noise = torch.randn(x.shape[0],mean.shape[1]).to(self.device)
        z = mean + torch.exp(0.5*log_var).mul(noise)
        return mean, log_var, z
        
    def decode(self,z):
        reconstructed_data = self.decoder(z)
        return reconstructed_data

    def loss(self, mean, log_var, sample_data, reconstructed_data, lambda_mmd=10):
        """
        Compute Info-VAE loss
        
        Args:
            mean: latent mean [batch_size, latent_dim]
            log_var: latent log variance [batch_size, latent_dim]
            sample_data: input data [batch_size, 156]
            reconstructed_data: reconstructed data [batch_size, 156]
            lambda_mmd: weight for MMD term
        
        Returns:
            total_loss: total Info-VAE loss
            recon_loss: reconstruction loss
            mmd_loss: MMD divergence term
        """
        # Flatten sample_data if not already flattened
        sample_data = sample_data.view(sample_data.size(0), -1)  # [128, 156]
        
        # 1. Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstructed_data, sample_data, reduction='mean')
        
        # 2. Generate samples from prior for MMD
        batch_size = mean.size(0)
        z_prior = torch.randn(batch_size, mean.size(1)).to(self.device)
        
        # Reparameterized samples
        z = mean + torch.exp(0.5 * log_var) * torch.randn_like(mean)
        
        # 3. MMD between z and prior
        mmd_loss = compute_mmd(z, z_prior)
        
        # Total loss = reconstruction + λ * MMD
        total_loss = recon_loss + lambda_mmd * mmd_loss
        
        return total_loss
    
    def generate(self,x: torch.Tensor):
        _, _, z = self.encode(x)
        reconstructed_data = self.decode(z)
        return reconstructed_data
    
def InfoVAE_train(model,optimizer,lambda_mmd=10):
    early_stop = 500
    cnt = 0
    min_loss = float('inf')
    for i in range(model.epoch):
        # Sample time indices of size equal to the batch size
        # From sefl.x_aug_sig
        time_indics = sample_indices(model.x_aug_sig.shape[0],model.batch_size,"cuda")
        sample_data = model.x_aug_sig[time_indics]
        # Encode 
        mean, log_var, z = model.encode(sample_data)
        # Decode
        reconstructed_data = model.decode(z)
        # Calculate loss
        loss = model.loss(mean,log_var,sample_data.view(model.batch_size,-1),reconstructed_data,lambda_mmd=lambda_mmd)
        # Backpropogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print loss
        if i%500==0:
            print("Epoch {} loss {}".format(i,loss.item()))
        # Early stop
        if loss.item()<min_loss:
            min_loss = loss.item()
            cnt = 0
        else:
            cnt += 1
            if cnt>early_stop:
                break