class BetaVAE(nn.Module):
    def __init__(self, x_aug_sig, epoch, batch_size, hidden_dims: List, device) -> None:
        super(BetaVAE, self).__init__()

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

    def loss(self, mean, log_var, sample_data, reconstructed_data, beta=1.0):
        """
        Compute β-VAE loss
        
        Args:
            mean: latent mean [batch_size, latent_dim]
            log_var: latent log variance [batch_size, latent_dim]
            sample_data: input data [batch_size, 156]
            reconstructed_data: reconstructed data [batch_size, 156]
            beta: weight for KL divergence term (β parameter)
        
        Returns:
            total_loss: total β-VAE loss
            recon_loss: reconstruction loss
            kl_loss: KL divergence term
        """
        # Flatten sample_data if not already flattened
        sample_data = sample_data.view(sample_data.size(0), -1)  # [128, 156]
        batch_size = sample_data.size(0)

        # 1. Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstructed_data, sample_data, reduction='sum') / batch_size
        
        # 2. KL divergence between q(z|x) and p(z)
        # Analytical KL divergence: D_KL(N(μ, σ^2) || N(0, 1))
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / batch_size
        
        # Total loss = reconstruction + β * KL
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss

    def generate(self,x: torch.Tensor):
        _, _, z = self.encode(x)
        reconstructed_data = self.decode(z)
        return reconstructed_data