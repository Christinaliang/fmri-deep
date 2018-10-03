import numpy as np
import torch.optim as optim

import data as d
import model_pixel as mp
import transform as t

def load_options(name):
    """Saves experiment options under names to load in train and test."""
    if name == 'pixel2d_atlas':
        idx = d.read_idx('../data/Template_files/'+
                         'Schaefer2018_400Parcels_7Networks_PCAordering.txt')
        tr = t.Transforms((t.Reorder(0, slice(1, None, None)), 
                           t.Reorder(0, idx),
                           t.ChannelDim()))
        # (img, lbl) is (1 x 1 x 400 x 200, 1 x 1 x 400 x 200)
        train = d.RestAtlasDataset('../data/train', tr)
        test = d.RestAtlasDataset('../data/test', tr)
        model = mp.PixelCNN
        optimizer = optim.Adam
    example = train[0][0]
    ch, shape = example.shape[0], np.array(example.shape[1:])
    model = model(ch, shape)
    return {'dataset': (train, test), 'model': model, 'optimizer': optimizer}
