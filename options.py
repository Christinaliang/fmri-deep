import numpy as np
import torch.optim as optim

import data as d
import model_blocks as mb
import model_transition as mt
import transform as t

def load_options(name):
    """Saves experiment options under names to load in train and test."""
    if name == 'rest_conn_nol1_rprop':
        tr = t.Transforms((t.ChannelDim(), t.Decimate(),
                           t.Transpose((4,0,1,2,3))), apply_to = 'label')
        train = d.SingleRestDataset('../data/train/NC01', tr)
        test = [] # no test set
        model = mt.TransitionFC
        optimizer = optim.Rprop # optim.Adam
        
        example = train[0][1]
        ch, shape = example.shape[1], np.array(example.shape[2:])
        model = model(ch, shape, ch, shape)
    optimizer = optimizer(model.parameters())
    return {'dataset': (train, test), 'model': model, 'optimizer': optimizer}
