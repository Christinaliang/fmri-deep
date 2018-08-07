import torch
import torch.nn as nn

import functions as f
import model_pixel as mp
import model_transition as mt

class VPN(nn.Module):
    """Generates 3d video data using an encoding and decoding step.
    Abstract class.
    """
    def __init__(self, state_ch, state_shape):
        super().__init__()
        self.state_ch, self.state_shape = state_ch, state_shape
        self.register_buffer('state', self.initial())
        
    def initial(self):
        return torch.zeros((1, self.state_ch) + f.int_tuple(self.state_shape))
        
    def init_hidden_state(self):
        """Resets hidden state to 0"""
        self.state = self.initial()
        if torch.cuda.is_available():
            self.state = self.state.cuda()
            
    def detach_hidden_state(self):
        """Keeps hidden state but detaches it from autograd graph"""
        self.state = self.state.detach()
        if torch.cuda.is_available():
            self.state = self.state.cuda()
            
    def encode(self, x, state):
        """Encode input and current state into new state"""
        raise NotImplementedError
        
    def decode(self, x, state):
        """Decode input and current state into output"""
        raise NotImplementedError
        
    def forward(self, x):
        self.state = self.encode(x, self.state)
        x = self.decode(x, self.state)
        return x

class WeightPixelNet(VPN):
    """Combines the WeightTransitionNet encoder and the PixelCNN decoder to
    generate resting state data from structural data.
    in: frame 0, frame 1, structure, state 0
        frames: f_ch, f_shape
        structure: x_ch, x_shape
        state: state_ch, f_shape
            
        WTN: (f_ch + state_ch, f_shape), x -> (state_ch, f_shape)
        pix: (f_ch, f_shape), (state_ch, f_shape) -> (3*mix, f_shape)
    out: Pr(frame 1)
    """
    def __init__(self, f_ch, f_shape, x_ch, x_shape, state_ch,
                 wtn_depth = 3, wtn_width = 16, pix_depth = 10, pix_mix = 5):
        super().__init__(state_ch, f_shape)
        self.wtn = mt.WeightTransitionNet(f_ch + state_ch, f_shape, x_ch,
                                          x_shape, state_ch, depth = wtn_depth,
                                          width = wtn_width)
        self.pix = mp.PixelCNN(f_ch, f_shape, state_ch, depth = pix_depth,
                               num_mix = pix_mix)
        
    def encode(self, x, state):
        f0, f1, img = x
        past_f = f.concat(f0, state)
        return self.wtn((past_f, img))
    
    def decode(self, x, state):
        f0, f1, img = x
        return self.pix((f1, state))