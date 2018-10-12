import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import model_blocks as b
import functions as f

class Transition(b.Forward):
    """Defines a general architecture that takes in time data and computes 
    the next time step from the current time step.
    """
    def __init__(self, steps = 'single'):
        """steps: 'single' - predict one time step. 
            'full' - predict entire time series from first frame.
        """
        super().__init__()
        self.steps = steps
        
    def forward(self, x):
        """x should be the full time series."""
        x = f.cycle_axes(x)
        frame = x[0]
        video = frame.unsqueeze(-1) # add time dim back in order to cat
        for t in range(x.shape[0] - 1):
            frame = self.next_frame(frame) # batch x ch x shape
            video = torch.cat((video, frame.unsqueeze(-1)), -1) # b x c x s x t
            if self.steps == 'single':
                frame = x[t + 1]
        return video
    
    def next_frame(self, frame):
        raise NotImplementedError
        
    def loss(self, output, label):
        raise NotImplementedError
        
class MarkovLinear(nn.Module):
    """Linear layer but with softmax applied to each column.
    Should be called with bias = False for it to be a transition matrix.
    """
    def __init__(self, size):
        super().__init__()
        print('P: ({}, {})'.format(size, size))
        self.weight = nn.Parameter(torch.Tensor(size, size))
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.shape[1])
        self.weight.data.uniform_(-stdv, stdv)
        
    def forward(self, x):
        P = F.softmax(self.weight, dim = 1)
        R = F.linear(x, P)
        return R
        
class MarkovChain(Transition):
    """Treats input vector as distribution over states of a markov chain and
    uses backprop to learn the transition matrix.
    """
    def __init__(self, ch, shape):
        super().__init__()
        shape = np.roll(shape, 1)[1:] # remove time axis from shape
        size = f.size(shape) * ch
        # self.P = MarkovLinear(size)
        self.P = nn.Linear(size, size) # include bias or not?
        self.criterion = nn.KLDivLoss()
        
    def next_frame(self, frame):
        # frame is vector of probabilities, 
        # but we want to work in the log prob domain
        shape = frame.shape
        frame = frame.view(shape[0], -1)
        frame = torch.log(torch.clamp(frame, min=1e-6)) # log probs
        frame = self.P(frame)
        return frame.view(shape)
    
    def loss(self, output, label):
        """Assumes output and label are probabilities, applies KL div."""
        # output = torch.clamp(output, min=1e-6)
        # output = torch.log(output)
        output = F.log_softmax(output)
        return self.criterion(output, label)