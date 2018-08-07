import torch
import torch.nn as nn
import torch.nn.functional as F

import functions as f
import model_blocks as b

"""
Implements architecures based on the idea of an encoding and decoding step.
"""

class Autoenc(nn.Module):
    """Implements a normal autoencoder with an encode and decode phase."""
    def __init__(self):
        super().__init__()
        
    def encode(self, x):
        raise NotImplementedError

    def decode(self, x):
        raise NotImplementedError

    def forward(self, x):
        x = self.encode(x)
        if self.training:
            return self.decode(x)
        else:
            return {'pred': self.decode(x), 'code': 
                x.transpose(1,-1).transpose(1,-2).transpose(1,-3)}
                
class VAE(nn.Module):
    """Implements VAE (https://arxiv.org/abs/1312.6114)
    Code is from https://github.com/pytorch/examples/tree/master/vae
    encode should take in x and output mu, logvar
    decode should take in sample and output label
    
    call model.train() before train and model.eval() before test
    """
    def __init__(self):
        super().__init__()
        
    def encode(self, x):
        raise NotImplementedError

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, x):
        raise NotImplementedError

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        if self.training:
            return self.decode(z), mu, logvar
        else:
            return {'pred': self.decode(z), 'logvar': logvar, 'code':
                z.transpose(1,-1).transpose(1,-2).transpose(1,-3)}
                
def loss_VAE(output, label):
    """Loss from Appendix B from VAE paper (https://arxiv.org/abs/1312.6114)
    Code is from https://github.com/pytorch/examples/tree/master/vae
    0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    replaced BCE with MSE
    """
    recon_x, mu, logvar = output
    x = label
    # x = x.view(x.shape[0], -1)
    # recon_x = recon_x.view(recon_x.shape[0], -1)
    # loss = F.binary_cross_entropy(recon_x, x, size_average=False)
    loss = F.mse_loss(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return loss + KLD

class BrainAutoenc(Autoenc):
    """
    Defines encoding and decoding with Residual blocks, downsampling
    via strided convolution, and upsampling via transposed convolution
    """
    def __init__(self, ch, shape, kernel = 3, down_r = 3, blocks_per = 3):
        super().__init__()
        in_ch = ch
        nn_ch = 16
        ax = (0, 1) # don't downsample 3rd spatial dim
        
        down_modules = [b.Block_7x7(ch, nn_ch, shape)]
        up_modules = []
        ch = nn_ch
        for _ in range(down_r):
            for _ in range(blocks_per - 1):
                down_modules.append(b.ResBlock(ch, shape, kernel))
            down_modules.append(b.UBlock(ch, shape, kernel, 'down', axes = ax))
            ch = ch * 2
            shape = f.halve(shape, ax)
        for _ in range(down_r):
            for _ in range(blocks_per - 1):
                up_modules.append(b.ResBlock(ch, shape, kernel))
            up_modules.append(b.UBlock(ch, shape, kernel, 'up', axes = ax))
            ch = ch // 2
            shape = f.double(shape, ax)
        up_modules.append(b.Block_7x7(ch, in_ch, shape))
        self.encode_module = b.MultiModule(down_modules)
        self.decode_module = b.MultiModule(up_modules)
    
    def encode(self, x):
        return self.encode_module(x)
    
    def decode(self, x):
        return self.decode_module(x)

class BrainVAE(VAE):
    """
    Defines encoding and decoding with Residual blocks, downsampling
    via strided convolution, and upsampling via transposed convolution
    """
    def __init__(self, ch, shape, kernel = 3, down_r = 3, blocks_per = 3):
        super().__init__()
        in_ch = ch
        nn_ch = 16
        ax = (0, 1) # don't downsample 3rd spatial dim
        
        down_modules = [b.Block_7x7(ch, nn_ch, shape)]
        up_modules = []
        ch = nn_ch
        for _ in range(down_r):
            for _ in range(blocks_per - 1):
                down_modules.append(b.ResBlock(ch, shape, kernel))
            down_modules.append(b.UBlock(ch, shape, kernel, 'down', axes = ax))
            ch = ch * 2
            shape = f.halve(shape, ax)
        self.mu_module = b.ResBlock(ch, shape, kernel)
        self.var_module = b.ResBlock(ch, shape, kernel)
        for _ in range(down_r):
            for _ in range(blocks_per - 1):
                up_modules.append(b.ResBlock(ch, shape, kernel))
            up_modules.append(b.UBlock(ch, shape, kernel, 'up', axes = ax))
            ch = ch // 2
            shape = f.double(shape, ax)
        up_modules.append(b.Block_7x7(ch, in_ch, shape))
        self.encode_module = b.MultiModule(down_modules)
        self.decode_module = b.MultiModule(up_modules)
    
    def encode(self, x):
        x = self.encode_module(x)
        return self.mu_module(x), self.var_module(x)
    
    def decode(self, x):
        return self.decode_module(x)