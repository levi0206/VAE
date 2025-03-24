import torch
from torch import nn
import torch.nn.functional as F
from lib.utils import sample_indices,compute_mmd
from lib.aug import sig_normal

class WAE(nn.Module):
    def __init__(self, x_aug_sig, epoch, batch_size, hidden_dims, device):
        super(WAE, self).__init__()

        self.x_aug_sig = x_aug_sig  
        self.epoch = epoch
        self.batch_size = batch_size
        self.device = device
        self.type = "WAE"
        self.loss_record = []
        
        self.encoder_mu = nn.Sequential(
            nn.Linear(hidden_dims[0],hidden_dims[1]),
            nn.LeakyReLU(),
            nn.Linear(hidden_dims[1],hidden_dims[2]),
            nn.LeakyReLU(),
        )
        self.encoder_sigma = nn.Sequential(
            nn.Linear(hidden_dims[0],hidden_dims[1]),
            nn.LeakyReLU(),
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
        x_flatten = x.view(x.shape[0], -1)
        mean = self.encoder_mu(x_flatten)
        log_var = self.encoder_sigma(x_flatten)
        # Clipping
        log_var = torch.clamp(log_var, min=-10, max=10)
        noise = torch.randn(x.shape[0], mean.shape[1]).to(self.device)
        z = mean + torch.exp(0.5 * log_var).mul(noise)
        return mean, log_var, z
        
    def decode(self, z):
        reconstructed_data = self.decoder(z)
        return reconstructed_data

    def loss(self, sample_data, reconstructed_data, z, lambda_mmd=10.0):
        """
        Compute WAE loss: reconstruction + MMD penalty
        
        Args:
            sample_data: input data [batch_size, sig_degree]
            reconstructed_data: reconstructed data [batch_size, sig_degree]
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
    early_stop = 400
    cnt = 0
    min_loss = float('inf')
    for i in range(model.epoch):
        # Sample batch
        time_indices = sample_indices(model.x_aug_sig.shape[0], model.batch_size, model.device)
        sample_data = model.x_aug_sig[time_indices]  # [batch_size, 39, 4]

        # Forward pass
        _,_,z = model.encode(sample_data)
        reconstructed_data = model.decode(z)
        reconstructed_data = sig_normal(reconstructed_data,True)

        # Compute loss
        loss = model.loss(sample_data, reconstructed_data, z, lambda_mmd=10.0)
        model.loss_record.append(loss.item())

        # Backpropogation
        optimizer.zero_grad()
        if loss<min_loss:
            loss.backward()
            optimizer.step()

        # Print loss
        if i%100==0:
            print("Epoch {} loss {:.2f}".format(i,loss.item()))
        # Early stop
        if loss.item()<min_loss:
            min_loss = loss.item()
            cnt = 0
        else:
            cnt += 1
            if cnt>early_stop:
                print("min_loss: {:.2f}".format(min_loss))
                break