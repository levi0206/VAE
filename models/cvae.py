from lib.utils import sample_indices
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from lib.utils import sample_indices

class CVAE(nn.Module):
    def __init__(self, x_aug_sig, x_original, epoch, batch_size, hidden_dims: List, latent_dim, device):
        super(CVAE, self).__init__()

        self.x_aug_sig = x_aug_sig  # Augmented input tensor [985, 39, 4]
        self.x_original = x_original  # Original input tensor, assuming shape [985, 39, 4]
        self.epoch = epoch
        self.batch_size = batch_size
        self.device = device
        input_dim = hidden_dims[0]  # 156 from your setup (39*4)

        # Condition dimension from x_original (flattened: 39*4 = 156)
        condition_dim = x_original.shape[1] * x_original.shape[2]  # 156

        # Encoder with condition
        self.encoder_fc = nn.Linear(input_dim + condition_dim, hidden_dims[1])  # 156 + 156 = 312
        self.encoder_mean = nn.Linear(hidden_dims[1], latent_dim)
        self.encoder_log_var = nn.Linear(hidden_dims[1], latent_dim)

        # Decoder with condition
        self.decoder_fc = nn.Linear(latent_dim + condition_dim, hidden_dims[1])  # latent_dim + 156
        self.decoder_out = nn.Linear(hidden_dims[1], input_dim)

        self.leaky_relu = nn.LeakyReLU()

        self.to(device)

    def encode(self, x, c):
        x_flatten = x.view(x.shape[0], -1)  # Flatten input [batch_size, 156]
        c_flatten = c.view(c.shape[0], -1)  # Flatten condition [batch_size, 156]

        x_concat = torch.cat([x_flatten, c_flatten], dim=1)  # Concatenate input and condition [batch_size, 312]
        hidden = self.leaky_relu(self.encoder_fc(x_concat))

        mean = self.encoder_mean(hidden)
        log_var = self.encoder_log_var(hidden)
        log_var = torch.clamp(log_var, min=-10, max=10)  # Prevent NaN issues

        noise = torch.randn_like(mean).to(self.device)
        z = mean + torch.exp(0.5 * log_var) * noise  # Reparameterization trick

        return mean, log_var, z

    def decode(self, z, c):
        c_flatten = c.view(c.shape[0], -1)  # Flatten condition [batch_size, 156]
        z_concat = torch.cat([z, c_flatten], dim=1)  # Concatenate latent space and condition

        hidden = self.leaky_relu(self.decoder_fc(z_concat))
        reconstructed_data = self.decoder_out(hidden)

        return reconstructed_data

    def loss(self, mean, log_var, sample_data, reconstructed_data):
        recon_loss = F.mse_loss(sample_data, reconstructed_data, reduction='sum')
        kl_loss = 0.5 * torch.sum(mean.pow(2) + log_var.exp() - 1 - log_var)
        beta = 0.001  # Scaling factor for KL loss
        return recon_loss + beta * kl_loss
    
    def generate(self, x: torch.Tensor, cond: torch.Tensor):
        """
        Args:
            x: input tensor [batch_size, 39, 4]
            cond: condition tensor [batch_size, 20, 1]
        """
        _, _, z = self.encode(x, cond)
        reconstructed_data = self.decode(z, cond)
        return reconstructed_data

def CAVE_train(model, optimizer):
    for i in range(model.epoch):
        # Sample same indices for both augmented and original data
        time_indices = sample_indices(model.x_aug_sig.shape[0], model.batch_size, model.device)
        sample_data = model.x_aug_sig[time_indices]  # [batch_size, 39, 4]
        condition_data = model.x_original[time_indices]  # [batch_size, 39, 4]

        # Forward pass
        mean, log_var, z = model.encode(sample_data, condition_data)
        reconstructed_data = model.decode(z, condition_data)

        # Compute loss
        loss = model.loss(mean, log_var, sample_data.view(model.batch_size, -1), reconstructed_data)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss
        if i % 10 == 0:
            print(f"Epoch {i}, Loss: {loss.item()}")