import torch
import torch.nn as nn

import model_blocks as b

class PixelCNN(b.Forward):
    """Stacks Pixel2d blocks together to take in an image and fit a logistic 
    mixture model.
    No dilations for now due to the assumption that temporal and physical
    dependencies are mostly local.
    loss function requires single batch, single channel.
    
    Input channels: channels in the image (for now, it must be 1)
    Output channels: log_prob, mean, and log_scale for each logistic (3 * nmix)
    """
    def __init__(self, ch, shape,
                 kernel = 3, depth = 20, nmix = 10):
        super().__init__()
        nn_ch = 16 # number of internal channels
        modules = [Pixel2d(nn_ch, shape, kernel, first = True, in_ch = ch)]
        for _ in range(depth):
            modules.append(Pixel2d(nn_ch, shape, kernel))
        modules.append(Pixel2d(3 * nmix, shape, kernel, in_ch = nn_ch))
        self.module = b.MultiModule(modules)
        
    def forward(self, x):
        h, v = self.module((x, x))
        return h
    
    def loss(self, output, label):
        return logistic_mixture_loss(output, label)
        
class Pixel2d(nn.Module):
    """Implements the PixelCNN block (https://arxiv.org/abs/1606.05328).
    first: determines whether to mask middle pixel or not
    """
    def __init__(self, ch, shape, kernel, 
                 dil = 1, first = False, in_ch = None):
        super().__init__()
        hconvf = HConv2d_A if first else HConv2d_B
        if in_ch is None: in_ch = ch
        self.skip = (in_ch == ch) # skip if in and out ch are the same
        self.hconv = b.conv_padded(in_ch, ch * 2, shape, shape, kernel,
                                   dil = dil, convf = hconvf)
        self.vconv = b.conv_padded(in_ch, ch * 2, shape, shape, kernel,
                                   dil = dil, convf = VConv2d)
        self.conv1x1 = b.conv_padded(ch * 2, ch * 2, shape, shape, 1)
        self.last = b.conv_padded(ch, ch, shape, shape, 1)  
        self.act = GateAct()
        
    def forward(self, x):
        """x is a tuple of the horizontal and vertical channels."""
        h, v = x
        h_initial = h
        
        v = self.vconv(v)
        v_shift = v.clone()
        v_shift.fill_(0)
        v_shift[:,:,:,1:] = v[:,:,:,:-1] # shift vertically for causality
        
        h = self.hconv(h) + self.conv1x1(v_shift)
        
        v, h = self.act(v), self.act(h)
        h = self.last(h)
        if self.skip: h += h_initial
        return (h, v)

class GateAct(nn.Module):
    """Implements the gated activation function in PixelCNN."""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """x should have an even # of channels."""
        C = x.shape[1]
        x1, x2 = x[:,:(C//2)], x[:,(C//2):]
        return torch.tanh(x1) * torch.sigmoid(x2)

def log_sum_exp(x, axis = 0):
    """Numerically stable log_sum_exp implementation
    Calculates log(sum(e^{x_i})).
    Axis should be the axis to sum over.
    """
    m, _  = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))

def log_softmax(x, axis = 0):
    """Numerically stable log_softmax implementation
    Converts vector into log probabilities.
    Axis should be the axis to convert.
    """
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), 
                                       dim=axis, keepdim=True))

def logistic_mixture_loss(output, label):
    """
    label: 1 x 1 x 64 x 64 x 38
    output: 1 x (3 * num_mix) x 64 x 64 x 38 - pi_i, mu_i, s_i
    example num_mix: 5
    assumes no rescaling on label.
    label is discretized to int.
    
    returns log probability of pixels, summed over whole image.
    Only works for single channel data.
    """
    nmix = int(output.shape[1] / 3)
    
    # All of theese are (mixtures x shape)
    logit_probs = output[0,:nmix]
    means = output[0,nmix:2*nmix]
    log_scales = torch.clamp(output[0,2*nmix:3*nmix], min=-7.)
    label = torch.round(label[0,0]).unsqueeze(0)
    label = torch.cat([label for _ in range(nmix)], 0)
    
    # bin reals into +- 0.5
    cdf_plus = torch.sigmoid(torch.exp(-log_scales) * (label - means + 0.5))
    cdf_minus = torch.sigmoid(torch.exp(-log_scales) * (label - means - 0.5))
    
    log_probs = torch.clamp(cdf_plus - cdf_minus, min=1e-12)
    log_probs_mix = log_probs + log_softmax(logit_probs)
    log_probs_max = log_sum_exp(log_probs_mix)
    return -torch.sum(log_probs_max)

class MaskedConv2d(nn.Conv2d):
    """Base class for causal convolutions. kernel size should be odd."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        self.mask.fill_(0)
    
    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

class HConv2d_A(MaskedConv2d):
    """Horizontal channel of Pixel, type A (excludes middle pixel)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _, _, kH, kW = self.weight.size()
        self.mask[:, :, kH//2, :kW//2] = 1

class HConv2d_B(MaskedConv2d):
    """Horizontal channel of Pixel, type B, (includes middle pixel)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _, _, kH, kW = self.weight.size()
        self.mask[:, :, kH//2, :(kW//2 + 1)] = 1

class VConv2d(MaskedConv2d):
    """Vertical channel of Pixel"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _, _, kH, kW = self.weight.size()
        self.mask[:, :, :(kH//2 + 1), :] = 1