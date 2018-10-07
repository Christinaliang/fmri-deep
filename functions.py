import numpy as np
from scipy.interpolate import interp2d
from skimage.measure import regionprops
import torch
import sys

"""
Defines general utility functions
"""
def cycle_axes(tensor):
    axes = np.arange(len(tensor.shape))
    axes = np.roll(axes, 1)
    return tensor.permute(int_tuple(axes))

def box(img, mask):
    bbox = regionprops(mask.astype(int), cache = False)[0].bbox
    return img[bbox_slice(bbox, len(img.shape))]

def bbox_slice(bbox, length):
    front, back = bbox[:len(bbox)//2], bbox[len(bbox)//2:]
    slices = []
    for i in range(len(front)):
        slices.append(slice(front[i], back[i]))
    for _ in range(length - len(front)):
        slices.append(slice(None, None))
    return slices

def islist(arr):
    return type(arr) in (tuple, list, np.array)

def halve(shape, axes):
    shape = shape.copy()
    for i in range(len(shape)):
        if i in axes:
            shape[i] = shape[i] / 2
    return shape

def double(shape, axes):
    shape = shape.copy()
    for i in range(len(shape)):
        if i in axes:
            shape[i] = shape[i] * 2
    return shape

def int_tuple(arr):
    return tuple([int(a) for a in arr])

def size(shape):
    i = 1
    for s in shape:
        i *= s
    return int(i)

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.filename = filename
    
    def write(self, message):
        self.terminal.write(message)
        with open(self.filename, "a+") as f:
            f.write(message)
            
    def flush(self):
        self.terminal.flush()
           
def to_boolean(s):
    return s.lower() in ('true', 'yes', 't', '1')

def pad_tuple(pad):
    padding = []
    for p in pad:
        padding.append(p // 2)
        padding.append(p // 2 + p % 2)
    padding = padding[::-1]
    return int_tuple(padding)

def padding(d_in, d_out, kernel, stride = 1, dil = 1):
    """Finds padding such that convolution outputs d_out given d_in.
    
    Padding is one-sided.
    out = (1 / stride)(in + 2 * padding - dilation * (kernel - 1) - 1) + 1
    """
    p = lambda i, o, k, s: max(0, int((o - 1) * s + dil*(k - 1) + 1 - i))
    return pad_tuple(apply_padf(d_in, d_out, kernel, stride, p))

def padding_t(d_in, d_out, kernel, stride = 1, dil = 1):
    """Finds the output padding such that conv_transpose outputs
    d_out given d_in.
    
    Output padding is one-sided.
    out = (in - 1) * stride + dilation * (kernel - 1) + 1
    """
    p = lambda i, o, k, s: max(0, int(o - ((i - 1) * s + dil*(k - 1) + 1)))
    return pad_tuple(apply_padf(d_in, d_out, kernel, stride, p))
    
def apply_padf(d_in, d_out, kernel, stride, pad_f):
    """Calculates padding based on arbitrary padding calculator.
    """
    if not islist(kernel):
        kernel = [kernel for _ in range(len(d_in))]
    if not islist(stride):
        stride = [stride for _ in range(len(d_in))]
    return np.array([pad_f(i, o, k, s) for i, o, k, s in 
                     zip(d_in, d_out, kernel, stride)])
        
def concat(a, b):
    """Concatenates a and b on the channel dimension.
    
    Assumes a and b are B x C x ....
    """
    return torch.cat((a, b), 1)

def tile_add(x, y):
    """Adds x and y with necessary tiling in the channel dimension.
    
    Assumes x and y are B x C x H x W (x D). Adds x and y 
    elementwise by tiling the smaller along the C dimension. 
    H, W, and D must be the same and the larger channel dimension
    must be a multiple of the smaller one.
    """
    if x.shape[1] > y.shape[1]:
        x, y = y, x
    ratio = y.shape[1] // x.shape[1]
    if len(x.shape) == 4:
        return x.repeat(1, ratio, 1, 1) + y
    return x.repeat(1, ratio, 1, 1, 1) + y
    
def double_arr_2d(arr, kind = 'zero'):
    """Given 2d array, upsamples by 2.
    
    kind: 'zero', 'linear', or any in scipy.interpolate.interp2d
    """
    if kind == 'zero':
        arr2 = np.zeros(np.array(arr.shape) * 2)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                arr2[2*i:2*i+2, 2*j:2*j+2] = arr[i,j]
        return arr2
    else:
        def double_length(x):
            return np.array(list(range(x * 2 - 1, x * (x * 2 - 1) 
                   + 1, x - 1))) / (x * 2 - 1)
        x = np.array(list(range(1, arr.shape[1] + 1)))
        y = np.array(list(range(1, arr.shape[0] + 1)))
        i = interp2d(x, y, arr, kind = kind)
        x2 = double_length(len(x))
        y2 = double_length(len(y))
        return i(x2, y2)
            
def double_weight_2d(arr, kind = 'zero'):
    """Given a 2d CNN weight, upsamples each kernel by 2.
    
    kind: 'zero', 'linear', or any in scipy.interpolate.interp2d
    weights are (Out, In, K_h, K_w)
    """
    nrr = np.zeros(np.array(arr.shape) * np.array((1,1,2,2)) 
                   + np.array((0,0,1,1)))
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            nrr[i][j] = double_arr_2d(arr[i][j], kind)
    return nrr