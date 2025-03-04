import torch
from torch import nn
import torch.nn.functional as F
from lib.utils import sample_indices,compute_mmd

class WAE(nn.Module):
    def __init__(self, x_aug_sig, epoch, batch_size, hidden_dims, latent_dim, device):
        super(WAE, self).__init__()

        self.x_aug_sig = x_aug_sig  # Input tensor [985, 39, 4]
        self.epoch = epoch
        self.batch_size = batch_size
        self.device = device
        input_dim = hidden_dims[0]  # 156 from your setup (39*4)

        # Encoder (deterministic, no variational parameters)
        self.encoder_fc = nn.Linear(input_dim, hidden_dims[1])  # 156 -> 100
        self.encoder_out = nn.Linear(hidden_dims[1], latent_dim)  # 100 -> 20

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dims[1])  # 20 -> 100
        self.decoder_out = nn.Linear(hidden_dims[1], input_dim)  # 100 -> 156

        self.leaky_relu = nn.LeakyReLU()

        self.to(device)

    def encode(self, x):
        x_flatten = x.view(x.shape[0], -1)  # Flatten input [batch_size, 156]
        hidden = self.leaky_relu(self.encoder_fc(x_flatten))
        z = self.encoder_out(hidden)  # Deterministic latent representation [batch_size, latent_dim]
        return z

    def decode(self, z):
        hidden = self.leaky_relu(self.decoder_fc(z))
        reconstructed_data = self.decoder_out(hidden)
        return reconstructed_data

    def loss(self, sample_data, reconstructed_data, z, lambda_mmd=10.0):
        """
        Compute WAE loss: reconstruction + MMD penalty
        
        Args:
            sample_data: input data [batch_size, 156]
            reconstructed_data: reconstructed data [batch_size, 156]
            z: latent representation [batch_size, latent_dim]
            lambda_mmd: weight for MMD term
        
        Returns:
            total_loss: total WAE loss
            recon_loss: reconstruction loss
            mmd_loss: MMD penalty
        """
        sample_data = sample_data.view(sample_data.size(0), -1)  # [batch_size, 156]

        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstructed_data, sample_data, reduction='mean') 

        # Sample from prior (standard normal)
        prior_samples = torch.randn_like(z).to(self.device)

        # MMD penalty
        mmd_loss = compute_mmd(z, prior_samples)

        # Total loss
        total_loss = recon_loss + lambda_mmd * mmd_loss
        
        return total_loss

def WAE_train(model, optimizer):
    early_stop = 600
    cnt = 0
    min_loss = float('inf')
    for i in range(model.epoch):
        # Sample batch
        time_indices = sample_indices(model.x_aug_sig.shape[0], model.batch_size, model.device)
        sample_data = model.x_aug_sig[time_indices]  # [batch_size, 39, 4]

        # Forward pass
        z = model.encode(sample_data)
        reconstructed_data = model.decode(z)

        # Compute loss
        loss = model.loss(sample_data, reconstructed_data, z, lambda_mmd=10.0)

        # Backpropagation
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