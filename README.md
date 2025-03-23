## log
- 0303: 
    - Able to run VAE using signature as input. However, the loss explodes.
    - Able to use mmd.py
    - Move VAE to VAE.py and be able to import VAE as a module
    - Now, I'm implementing InfoVAE. The MMD loss function remains to be fixed.
- 0304:
    - Implement InfoVAE, BetaVAE, CVAE, WAE.
    - The models are all able to run successfully.
- 0308:
    - It seems that the WAE with signature kernel is not feasible.
- 0322 (21:25):
    - Fix the explosion of loss. The reason is due to incorret implementation. BetaVAE becomes normal.
    - Complete the data example.
    - Update the ``mmd_loss`` so that the function accepts signatures of tensor shape ``[batch,dim]``.
    - Update ``sig_mmd.py``.
    - Set seed.
- 0323:
    - Complete a test for betavae. 
    - Change all activation functions to ``nn.LeakyReLU()``.
    - Use ``nn.KLDivLoss(reduction="batchmean")`` instead of self-implemented function.
    - Use JD divergence instead of KL divergence.
    - Fix training loop.
    - We will use VAE, Beta-VAE, Info-VAE, and WAE-MMD for our experiments. 
    - Complete the full implementation for signature degree 3, 4, and 5.
    - Complete the numerical experiments!