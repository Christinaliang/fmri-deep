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
        x = x.cycle_axes(x)
        frame = x[0]
        video = frame
        for t in range(x.shape[0] - 1):
            video = np.concatenate((video, self.next_frame(frame)), axis = 0)
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
        self.register_buffer('weights', torch.ones((size,size)))
        
    def forward(self, x):
        return F.linear(x, F.softmax(self.weights, dim = 1))
        
class MarkovChain(Transition):
    """Treats input vector as distribution over states of a markov chain and
    uses backprop to learn the transition matrix.
    """
    def __init__(self, ch, shape):
        super().__init__()
        size = f.size(shape) * ch
        self.P = MarkovLinear(size)
        self.criterion = nn.KLDivLoss()
        
    def next_frame(self, frame):
        shape = frame.shape
        frame = frame.view(shape[0], -1)
        frame = self.P(frame)
        return frame.view(shape)
    
    def loss(self, output, label):
        """Assumes output and label are probabilities, applies KL div."""
        output = torch.log(output) # modify to make numerically stable?
        return self.criterion(output, label)