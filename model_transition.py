import numpy as np
import torch
import torch.nn as nn

import model_blocks as mb

class Transition(nn.Module):
    """Defines a general architecture that takes in time data and computes 
    the next time step from the current time step.
    """
    def __init__(self):
        super().__init__()
        
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
    
    def compute_loss(self, image, label):
        raise NotImplementedError

class TransitionFC(Transition):
    """Predicts transition via fully connected layer without activation.
    """
    def __init__(self, in_ch, in_shape, out_ch, out_shape, l1 = 0):
        super().__init__()
        self.fc = mb.FC(in_ch, in_shape, out_ch, out_shape)
        self.l1 = l1 # lambda value for l1 regularization on FC matrix
        self.loss_func = nn.MSELoss()
    
    def forward(self, x):
        return self.fc(x)
    
    def compute_loss(self, image, label):
        if next(self.parameters()).is_cuda():
            image, label = image.cuda(), label.cuda()
        output = self.forward(image)
        return self.loss_func(output, label) + self.l1 * l1reg(self)

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
            if W.is_cuda():
                target = target.cuda()
            reg += l1_f(W, target)
    return reg
