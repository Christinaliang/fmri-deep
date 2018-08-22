import numpy as np
import torch
import torch.nn as nn

import functions as f

"""
The model modules 3 layers of abstraction: the operations themselves (ex: 
conv_padded, FC), blocks that combine operations (ex: ResBlock, UBlock), and
the full models that combine blocks (found in model_autoenc and others).

Notes:
    shapes should be np.arrays of floats for nice down and upsampling:
        7.0 -> 3.5 -> 7.0 vs 7 -> 3 -> 6
"""

class FC(nn.Module):
    """Implements a fully connected layer that reshapes before and after."""
    def __init__(self, in_ch, in_shape, out_ch, out_shape):
        super().__init__()
        in_shape = np.array(in_shape).astype(int)
        out_shape = np.array(out_shape).astype(int)
        
        self.out_ch, self.out_shape = out_ch, f.int_tuple(out_shape)
        self.fc = nn.Linear(in_ch * f.size(in_shape), 
                            out_ch * f.size(out_shape))
        
    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.fc(x)
        x = torch.reshape(x, (x.shape[0], self.out_ch) + self.out_shape)
        return x

class Block(nn.Module):
    """Implements a generic conv-BN-ReLU block."""
    def __init__(self, ch, shape, kernel, dil = 1):
        super().__init__()
        self.module = MultiModule((
                conv_padded(ch, ch, shape, shape, kernel, dil = dil),
                batch(ch, shape), nn.ReLU()))
    
    def forward(self, x):
        return self.module(x)

class Block_7x7(nn.Module):
    """Implements a block intended to be the first block in a network.
    Characterized by large kernel, increase in channels, and no skip.
    """
    def __init__(self, in_ch, out_ch, shape, kernel = 7):
        super().__init__()
        self.module = MultiModule((
                conv_padded(in_ch, out_ch, shape, shape, kernel),
                batch(out_ch, shape), nn.ReLU()))
    
    def forward(self, x):
        return self.module(x)

class Printer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        print(x.shape)
        return x

class UBlock(nn.Module):
    """Implements a block inspired by UNet (https://arxiv.org/abs/1505.04597)
    and VNet (https://arxiv.org/abs/1606.04797).
    kernel: the kernel size of the res block (not the downsampling)
    direction: either 'up' or 'down'
        down: shape is cut in half and ch is doubled after the block.
        up: shape is doubled and ch is cut in half
    axes: the axes to downsample (ex: (0, 1, 2)). None means downsample all.
        
    shortcut connections between downsampling layers and upsampling layers are
    also recommended (as described in UNet) but not implemented in this block.
    """
    def __init__(self, ch, shape, kernel, direction, length = 2, axes = None):
        super().__init__()
        module_list = []
        for _ in range(length):
            module_list.append(Block(ch, shape, kernel))
        self.module1 = MultiModule(module_list)
        self.module2 = StrideBlock(ch, shape, direction, axes = axes)
        
    def forward(self, x):
        x = x + self.module1(x)
        return self.module2(x)
    
class StrideBlock(nn.Module):
    """Implements strided convolution with batchnorm and relu. Down/upsampled
    axes are selectable.
    """
    def __init__(self, ch, shape, direction, axes = None):
        super().__init__()
        if axes is None:
            s = 2
            axes = [i for i in range(len(shape))]
        else:
            s = f.int_tuple([1 + (i in axes) for i in range(len(shape))])
        module_list = []
        if direction == 'down':
            out_shape = f.halve(shape, axes)
            module_list.append(conv_padded(ch, ch * 2, shape, out_shape, 
                                           s, stride = s))
            ch = ch * 2
            shape = out_shape
        elif direction == 'up':
            out_shape = f.double(shape, axes)
            module_list.append(conv_padded_t(ch, ch // 2, shape, out_shape,
                                             s, stride = s))
            ch = ch // 2
            shape = out_shape
        module_list.extend((batch(ch, shape), nn.ReLU()))
        self.module = MultiModule(module_list)
        
    def forward(self, x):
        return self.module(x)

class ResBlock(nn.Module):
    """Implements the residual block:
    (http://torch.ch/blog/2016/02/04/resnets.html)
    """
    def __init__(self, ch, shape, kernel, dil = 1):
        super().__init__()
        self.module = MultiModule((
                Block(ch, shape, kernel, dil = dil), 
                conv_padded(ch, ch, shape, shape, kernel, dil = dil), 
                batch(ch, shape)))
    
    def forward(self, x):
        return x + self.module(x)
    
def batch(ch, shape):
    if len(shape) == 2:
        return nn.BatchNorm2d(ch)
    elif len(shape) == 3:
        return nn.BatchNorm3d(ch)
    
def conv_padded(ch_in, ch_out, d_in, d_out, kernel, 
                stride = 1, dil = 1, convf = (nn.Conv2d, nn.Conv3d)):
    """Returns a padded conv module that takes in ch_in, d_in and
    outputs ch_out, d_out.
    
    conv_f: either a tuple specifying the convolution module for 2d and 3d
        or a single module (not a tuple).
    """
    d_in, d_out = f.int_tuple(d_in), f.int_tuple(d_out)
    if len(d_in) == 2:
        pad_f = nn.ConstantPad2d
        if f.islist(convf):
            convf = convf[0]
    elif len(d_in) == 3:
        pad_f = nn.ConstantPad3d
        if f.islist(convf):
            convf = convf[1]
    p = f.padding(d_in, d_out, kernel, stride)
    print(ch_in, d_in, ch_out, d_out)
    # print('pad:', p, 'kernel:', kernel, 'stride:', stride)
    conv = convf(ch_in, ch_out, kernel, stride = stride)
    pad = pad_f(p, 0)
    return MultiModule([pad, conv])

def conv_padded_t(ch_in, ch_out, d_in, d_out, kernel, 
                  stride = 1, dil = 1, 
                  convf = (nn.ConvTranspose2d, nn.ConvTranspose3d)):
    """Returns a padded transposed conv module that takes in 
    ch_in, d_in and outputs ch_out, d_out.
    
    conv_f: either a tuple specifying the convolution module for 2d and 3d
        or a single module (not a tuple).
    """
    d_in, d_out = f.int_tuple(d_in), f.int_tuple(d_out)
    if len(d_in) == 2:
        pad_f = nn.ConstantPad2d
        if f.islist(convf):
            convf = convf[0]
    elif len(d_in) == 3:
        pad_f = nn.ConstantPad3d
        if f.islist(convf):
            convf = convf[1]
    p = f.padding_t(d_in, d_out, kernel, stride)
    print(ch_in, d_in, ch_out, d_out)
    # print('pad:', p, 'kernel:', kernel, 'stride:', stride)
    conv = convf(ch_in, ch_out, kernel, stride = stride)
    pad = pad_f(p, 0)
    return MultiModule([conv, pad])

class MultiModule(nn.Module):
    """Composes modules and one-arg functions into one module."""
    def __init__(self, module_list):
        super().__init__()
        self.module_list = module_list
        for i, module in enumerate(self.module_list):
            if isinstance(module, nn.Module):
                self.add_module("module_" + str(i), module)
    
    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        return x