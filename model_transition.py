import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import functions as f
import model_blocks as b

"""
Implements models that take in two inputs, one of which evolves over time and
the other of which predicts how the first evolves over time.
"""
class TransitionNet(nn.Module):
    """Defines a general architecture where static data is encoded and
    this code is used in a transition step on dynamic data.
    """
    def __init__(self):
        super().__init__()
    
    def encode(self, x):
        raise NotImplementedError
    
    def transition(self, x, code):
        raise NotImplementedError
    
    def forward(self, x):
        x_i, x_w = x
        code = self.encode(x_w)
        return self.transition(x_i, code)

class WeightTransitionNet(TransitionNet):
    """Uses structural information to predict the transitions of resting state
    data. Structure is encoded via UNet-style architecture and transitions
    with stacked residual blocks.
    in: frame t, structure
    out: frame t+1
    
    in -> out: (i_ch, f_shape), (s_ch, s_shape) -> (out_ch, f_shape)
    
    i_ch, i_shape: ch and shape of the frames
    s_ch, s_shape: ch and shape of the static image
    
    example:
        net = RestTransitionNet(1, (64, 64, 38), 6, (256, 256, 68), 3)
        use 3x3x3 kernels to transition from rest_t to rest_t+1
        rest is B x 1 x 64 x 64 x 38
        structure is B x 6 x 256 x 256 x 68
    """
    def __init__(self, i_ch, i_shape, s_ch, s_shape, o_ch, 
                 kernel = 3, depth = 3, width = 16):
        super().__init__()
        i_shape, s_shape = np.array(i_shape), np.array(s_shape)
        if not f.islist(kernel):
            kernel = [kernel for _ in range(len(i_shape))]
        
        def should_downsample(current, target):
            current = current // 2
            return [(c >= t) for c, t in zip(current, target)]
        
        ### encoding architecture
        # keep halving structure until it is almost kernel size, then use
        # a single unpadded convolution to equalize the sizes.
        # double channels every time dimensions are halved.
        # final channel count is code_ch.
        en_ch = 16
        modules = [b.Block_7x7(s_ch, en_ch, s_shape)]
        ch, shape = en_ch, s_shape
        status = should_downsample(shape, kernel)
        while True in status:
            axes = np.where(np.array(status).astype(int) == 1)[0]
            modules.append(b.StrideBlock(ch, shape, 'down', axes = axes))
            # TODO: channel stoppage?
            ch = ch * 2 # instead of just multiplying by 2, stop at nn_ch^2
            shape = f.halve(shape, axes)
            status = should_downsample(shape, kernel)
        # out = in - diff_kernel + 1
        diff_kernel = f.int_tuple(shape.astype(int) - np.array(kernel) + 1)
        modules.append(b.conv_padded(ch, ch, shape, kernel, diff_kernel))
        code_ch = ch
        self.encode_modules = b.MultiModule(modules)
        
        ### transition architecture
        # Uses code from encoding architecture to predict weights
        # TODO: add dilation?
        nn_ch = width
        self.pre = b.Block_7x7(i_ch, nn_ch, i_shape)
        modules = []
        for _ in range(depth):
            modules.append(WeightResBlock(nn_ch, i_shape, kernel, code_ch))
        self.mid = b.MultiModule(modules)
        self.post = b.Block_7x7(nn_ch, o_ch, i_shape)
    
    def encode(self, x):
        return self.encode_modules(x) # code is now B x C x kH x kW x kD
    
    def transition(self, x, code):
        x = self.pre(x)
        x, code = self.mid((x, code))
        x = self.post(x)
        return x

class WeightConv(nn.Module):
    """Takes in two inputs and uses the first to predict weights to convolve
    with the second.
    Input for weights: B x C x H x W x D
    Weight: C_out x C_in x kH x kW x kD
    Second Input: B x C_in x H x W x D
    
    For now, batch size must be 1.
    """
    def __init__(self, in_shape, out_shape, stride, dil):
        super().__init__()
        self.stride = stride
        self.dil = dil
        self.in_shape = in_shape
        self.out_shape = out_shape
        
    def gen_weights(self, x):
        """Should generate weights from x and reshape into 
        C_out x C_in x kH x kW x kD
        """
        raise NotImplementedError
        
    def forward(self, x):
        x_i, x_w = x # x_i is the dynamic input and x_w is the static data
        weights = self.gen_weights(x_w)
        x_i = Fconv_padded(x_i, weights, self.in_shape, self.out_shape,
                           self.stride, self.dil)
        return (x_i, x_w)
    
class WeightConv1x1(WeightConv):
    """Generates weights via 1x1 convolution. The x that generates weights
    should have shape B x C x kH x kW x kD (its shape must match kernel but
    channel # can be different).
    """
    def __init__(self, in_ch, out_ch, in_shape, out_shape, kernel, code_ch,
                 stride = 1, dil = 1):
        super().__init__(in_shape, out_shape, stride, dil)
        if not f.islist(kernel):
            kernel = [kernel for _ in range(len(in_shape))]
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel = kernel
        self.conv = b.conv_padded(code_ch, in_ch * out_ch, kernel, kernel, 1)
    
    def gen_weights(self, x):
        x = self.conv(x)
        x = torch.reshape(x, (self.out_ch, self.in_ch) + 
                          f.int_tuple(self.kernel))
        return x
    
class WeightResBlock(nn.Module):
    """Implements residual block but with WeightConv."""
    def __init__(self, ch, shape, kernel, code_ch, dil = 1,
                 convf = WeightConv1x1):
        super().__init__()
        self.module = b.MultiModule((
                WeightBlock(ch, shape, kernel, code_ch, 
                            dil = dil, convf = convf),
                convf(ch, ch, shape, shape, kernel, code_ch, dil=dil)))
        self.post = b.batch(ch, shape)
        
    def forward(self, x):
        x_init, _ = x
        x_i, x_w = self.module(x)
        x_i = x_init + self.post(x_i)
        return (x_i, x_w)

class WeightBlock(nn.Module):
    """Implements conv-BN-ReLU block but with WeightConv."""
    def __init__(self, ch, shape, kernel, code_ch, dil = 1,
                 convf = WeightConv1x1):
        super().__init__()
        self.module = convf(ch, ch, shape, shape, kernel, code_ch, dil=dil)
        self.post = b.MultiModule((b.batch(ch, shape), nn.ReLU()))
        
    def forward(self, x):
        x_i, x_w = self.module(x)
        x_i = self.post(x_i)
        return (x_i, x_w)

def Fconv_padded(x, weight, d_in, d_out, stride, dil):
    """
    x: B x C_in x H x W x D
    weight: C_out x C_in x kH x kW x kD
    """
    d_in, d_out = f.int_tuple(d_in), f.int_tuple(d_out)
    if len(d_in) == 2:
        convf = F.conv2d
    else:
        convf = F.conv3d
    kernel = f.int_tuple(weight.shape[2:]) # (kH, kW, kD)
    p = f.padding(d_in, d_out, kernel, stride = stride, dil = dil)
    return convf(F.pad(x, p), weight, stride = stride, dilation = dil)

def Fconv_padded_t(x, weight, d_in, d_out, stride, dil):
    """
    x: B x C_in x H x W x D
    weight: C_out x C_in x kH x kW x kD
    """
    d_in, d_out = f.int_tuple(d_in), f.int_tuple(d_out)
    if len(d_in) == 2:
        convf = F.conv_transpose2d
    else:
        convf = F.conv_transpose3d
    kernel = f.int_tuple(weight.shape[2:]) # (kH, kW, kD)
    p = f.padding_t(d_in, d_out, kernel, stride = stride, dil = dil)
    return F.pad(convf(x, weight, stride = stride, dilation = dil), p)