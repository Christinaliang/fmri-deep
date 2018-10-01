import numpy as np
import torch.optim as optim

import data as d
import model_blocks as mb
import model_transition as mt
import transform as t

def load_options(name):
    """Saves experiment options under names to load in train and test."""
    if name == 'markov_atlas':
        tr = t.Transforms((t.ChannelDim(), 
                           t.Transpose((2,0,1))), apply_to = 'label')
        train = d.RestVectorDataset('../data/train/NC01', tr)
        test = [] # no test set
        model = mt.TransitionFC
        optimizer = optim.Adam
        
        example = train[0][1]
        ch, shape = example.shape[1], np.array(example.shape[2:])
        model = model(ch, shape)
    return {'dataset': (train, test), 'model': model, 'optimizer': optimizer}
