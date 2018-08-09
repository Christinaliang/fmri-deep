import torch
import torch.nn as nn
import torch.nn.functional as F

import model_blocks as b

"""
Implements autoregressive models.
"""

class PixelCNN(nn.Module):
    """Stacks Pixel3d blocks together to make a Pixel decoder, which takes
    in an image and conditional information to output the probability of the
    image.
    in -> out: (f_ch, f_shape), (cond_ch, f_shape) -> (3*mix, f_shape)
    
    TODO: add dilation, maybe batch norm?
    assumes 1 channel in image (until act function is extended)
    """
    def __init__(self, ch, shape, cond_ch, 
                 kernel = 3, depth = 10, num_mix = 5):
        super().__init__()
        nn_ch = 16
        modules = [Pixel3d(nn_ch, shape, kernel, cond_ch, 
                           first = True, in_ch = ch)]
        for _ in range(depth): # actual depth is depth + 2
            modules.append(Pixel3d(nn_ch, shape, kernel, cond_ch))
        modules.append(Pixel3d(3 * num_mix, shape, kernel, cond_ch, 
                               in_ch = nn_ch))
        self.module = b.MultiModule(modules)
        
    def forward(self, x):
        img, cond = x
        h, v, d, cond = self.module((img, img, img, cond))
        return h

class Pixel3d(nn.Module):
    """Implements one block of 3d PixelCNN (https://arxiv.org/abs/1606.05328)
    first: whether this is the first block in the cnn or not. first block has
        masking that excludes the middle pixel, while non-first blocks include
        the middle pixel.
    """
    def __init__(self, ch, shape, kernel, cond_ch, dil = 1, first = False,
                 in_ch = None):
        super().__init__()
        if first:
            hconvf = HConv3d_A
        else:
            hconvf = HConv3d_B
        if in_ch is None:
            in_ch = ch
        self.skip = (in_ch == ch) # only skip if channels stay the same
        self.hconv = b.conv_padded(in_ch, ch*2, shape, shape, kernel, dil=dil,
                                 convf = hconvf)
        self.hact = GateAct(shapes = (cond_ch, ch, shape))
        self.vconv = b.conv_padded(in_ch, ch*2, shape, shape, kernel, dil=dil,
                                 convf = VConv3d)
        self.vact = GateAct(shapes = (cond_ch, ch, shape))
        self.dconv = b.conv_padded(in_ch, ch*2, shape, shape, kernel, dil=dil,
                                 convf = DConv3d)
        self.dact = GateAct(shapes = (cond_ch, ch, shape))
        self.d_to_v = b.conv_padded(ch*2, ch*2, shape, shape, 1)
        self.v_to_h = b.conv_padded(ch*2, ch*2, shape, shape, 1)
        self.last = b.conv_padded(ch, ch, shape, shape, 1)
        
    def forward(self, x):
        """Figure 2 of Gated PixelCNN (https://arxiv.org/abs/1606.05328)
        x: tuple of the horizontal, vertical, and depth-wise sections
        cond: conditional information- we want to estimate p(x | cond)
        """
        h, v, d, cond = x
        h_i = h
        d = self.dconv(d)
        d_shift = d.clone()
        d_shift.fill_(0)
        d_shift[:,:,:,:,1:] = d[:,:,:,:,:-1] # shift depth-wise for causality
        v = self.vconv(v) + self.d_to_v(d_shift)
        v_shift = v.clone()
        v_shift.fill_(0)
        v_shift[:,:,:,1:,:] = v[:,:,:,:-1,:] # shift vertically for causality
        h = self.hconv(h) + self.v_to_h(v_shift)
        
        d = self.dact(d, cond = cond)
        v = self.vact(v, cond = cond)
        h = self.hact(h, cond = cond)
        if self.skip:
            h = h_i + self.last(h) # skip connection
        return (h, v, d, cond)
    
class GateAct(nn.Module):
    """Implements the conditional gated activation function in PixelCNN.
    shapes:
        cond_ch: channels of condition matrix
        ch: 1/2 channels of x
        shape: shape of condition matrix, which should be the same as x
    Possible modfications: 
        instead of 1x1 convolutions for condition, use more elaborate encoding
    """
    def __init__(self, shapes = None):
        super().__init__()
        if shapes is not None:
            cond_ch, ch, shape = shapes
            self.conv_forget = b.conv_padded(cond_ch, ch, shape, shape, 1)
            self.conv_gate = b.conv_padded(cond_ch, ch, shape, shape, 1)
    
    def forward(self, x, cond = None):
        """x should have even # of channels."""
        _, C, _, _, _ = x.shape
        x1, x2 = x[:,:(C//2),:,:,:], x[:,(C//2):,:,:,:]
        if cond is not None:
            x1 = x1 + self.conv_forget(cond)
            x2 = x2 + self.conv_gate(cond)
        return torch.tanh(x1) * torch.sigmoid(x2)

def log_sum_exp(x, axis = 1):
    """Numerically stable log_sum_exp implementation
    Chooses the most likely mixture component.
    """
    m, _  = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))

def log_softmax(x, axis = 1):
    """Numerically stable log_softmax implementation
    Converts vector into log probabilities.
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
    TODO: make this functional for multichannel data
    """
    num_mix = int(output.shape[1] / 3)
    logit_probs = output[:,:num_mix,:,:,:] # 1 x 5 x shape
    means = output[:,num_mix:2*num_mix,:,:,:] # 1 x 5 x shape
    log_scales = torch.clamp(output[:,2*num_mix:3*num_mix,:,:,:], 
                             min=-7.) # 1 x 5 x shape, clamped to [-7,)
    label = torch.round(label)
    label = torch.cat([label for _ in range(num_mix)], 1) # 1 x 5 x shape
    cdf_plus = torch.sigmoid(torch.exp(-log_scales) * (label - means + 0.5))
    cdf_minus = torch.sigmoid(torch.exp(-log_scales) * (label - means - 0.5))
    
    log_probs = torch.clamp(cdf_plus - cdf_minus, min=1e-12)
    log_probs_mix = log_probs + log_softmax(logit_probs) # 1 x 5 x shape
    log_probs_max = log_sum_exp(log_probs_mix) # 1 x 1 x S: chooses 1 logistic
    return -torch.sum(log_probs_max)

class MaskedConv3d(nn.Conv3d):
    """Base class for causal convolutions. kernel size should be odd."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        self.mask.fill_(0)
    
    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

class HConv3d_A(MaskedConv3d):
    """Horizontal channel of Pixel, type A (excludes middle pixel)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _, _, kH, kW, kD = self.weight.size()
        self.mask[:, :, kH//2, :kW//2, kD//2] = 1

class HConv3d_B(MaskedConv3d):
    """Horizontal channel of Pixel, type B, (includes middle pixel)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _, _, kH, kW, kD = self.weight.size()
        self.mask[:, :, kH//2, :(kW//2 + 1), kD//2] = 1

class VConv3d(MaskedConv3d):
    """Vertical channel of Pixel"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _, _, kH, kW, kD = self.weight.size()
        self.mask[:, :, :(kH//2 + 1), :, kD//2] = 1

class DConv3d(MaskedConv3d):
    """Depth-wise channel of Pixel"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _, _, kH, kW, kD = self.weight.size()
        self.mask[:, :, :, :, :(kD//2 + 1)] = 1
