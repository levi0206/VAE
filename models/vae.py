import torch
from torch import nn
import torch.nn.functional as F
from typing import List
from lib.utils import sample_indices

class VAE(nn.Module):
    def __init__(self, x_aug_sig, epoch, batch_size, device, hidden_dims: List) -> None:
        super(VAE, self).__init__()

        print("Input tensor shape: {}".format(x_aug_sig.shape))
        print("Hidden dims: {}".format(hidden_dims))

        self.x_aug_sig = x_aug_sig
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

    def loss(self,mean,log_var,sample_data,reconstructed_data):
        # Reconstruction loss 
        recon_loss = F.mse_loss(sample_data, reconstructed_data, reduction='mean')
        # print(recon_loss.item())
        # KL divergence
        kl_loss = 0.5 * ((mean.pow(2) + log_var.exp() - 1 - log_var).mean()).sum()
        # Total VAE loss
        loss = recon_loss + kl_loss
        return loss
    
    def generate(self,x: torch.Tensor):
        _, _, z = self.encode(x)
        reconstructed_data = self.decode(z)
        return reconstructed_data
    
def VAE_train(model,optimizer):
    early_stop = 400
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
        loss = model.loss(mean,log_var,sample_data.view(model.batch_size,-1),reconstructed_data)

        # Backpropogation
        optimizer.zero_grad()
        if loss<min_loss:
            loss.backward()
            optimizer.step()

        # Print loss
        if i%100==0:
            print("Epoch {} loss {:4f}".format(i,loss.item()))

        # Early stop
        if loss.item()<min_loss:
            min_loss = loss.item()
            cnt = 0
        else:
            cnt += 1
            if cnt>early_stop:
                print("min_loss: {:4f}".format(min_loss))
                break