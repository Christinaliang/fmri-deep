import numpy as np
import torch
import torch.nn as nn

import model_blocks as b

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
            if torch.cuda.is_available():
                target = target.cuda()
            reg += l1_f(W, target)
    return reg
