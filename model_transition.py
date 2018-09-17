import numpy as np
import torch
import torch.nn as nn

import model_blocks as mb
import functions as f

class Transition(nn.Module):
    """Defines a general architecture that takes in time data and computes 
    the next time step from the current time step.
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()
        
    def forward(self, x):
        raise NotImplementedError
            
    def step(self, image, label, optimizer = None):
        avg_loss = 0.0
        label = label[0] # No batching
        for t in range(label.shape[0] - 1):
            frame_in = label[t].unsqueeze(0)
            frame_out = label[t+1].unsqueeze(0)
            
            if self.training:
                optimizer.zero_grad()
                loss = self.compute_loss(frame_in, frame_out)
                loss.backward()
                optimizer.step()
            else:
                loss = self.compute_loss(frame_in, frame_out)
            
            avg_loss += ((loss.data.item() - avg_loss) / (t + 1))
        return avg_loss
    
    def simulate(self, frame_0, time):
        frame = frame_0.unsqueeze(0)
        video = frame.detach()
        for t in range(time):
            frame = self.forward(frame).detach()
            video = np.concatenate((video, frame), axis = 0)
        return video
    
    def compute_loss(self, image, label):
        if next(self.parameters()).is_cuda:
            image, label = image.cuda(), label.cuda()
        output = self.forward(image)
        return self.loss_func(output, label)
    
    def loss_func(self, output, label):
        return self.criterion(output, label)

class MarkovLinear(nn.Linear):
    """Linear layer except the columns sum to one (models a Markov chain).
    Must be called with bias = False for it to make sense.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        print(self.weight.data.shape)
        self.mask.fill_(1)
        self.mask = self.mask - torch.eye(self.weight.data.shape[0])
        
    def forward(self, x):
        self.weight.data *= self.mask
        # weight is out_features x in_features
        # everything out of something should sum to 1
        col_totals = torch.sum(self.weight.data, 0)
        diag = torch.ones(col_totals.shape) - col_totals
        return super().forward(x) + diag * x

class TransitionMarkov(Transition):
    def __init__(self, ch, shape):
        super().__init__()
        self.ch = ch
        self.shape = f.int_tuple(shape)
        size = f.size(shape) * ch
        self.markov = MarkovLinear(size, size, bias = False)
        
    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.markov(x)
        x = torch.reshape(x, (x.shape[0], self.ch) + self.shape)
        return x

class TransitionFC(Transition):
    """Predicts transition via fully connected layer without activation.
    """
    def __init__(self, ch, shape, l1 = 0):
        super().__init__()
        self.fc = mb.FC(ch, shape, ch, shape)
        self.l1 = l1 # lambda value for l1 regularization on FC matrix
    
    def forward(self, x):
        return self.fc(x)
    
    def loss_func(self, output, label):
        reg = self.l1 * l1reg(self)
        return super().loss_func(output, label) + reg

def l1reg(model):
    """Computes the L1 norm of the weights of the model.
    TODO: add truncation?
        Compute which weights will cross zero after applying the L1 penalty
        Apply the proposed update to all weights
        Fill with zero the weights annotated as zero in step (1)
    """
    l1_f = nn.L1Loss(size_average = False)
    reg = 0
    for name, W in model.named_parameters():
        if not name.endswith('bias'):
            target = torch.zeros(W.shape)
            if W.is_cuda:
                target = target.cuda()
            reg += l1_f(W, target)
    return reg
