import numpy as np
import torch.optim as optim

import data as d
import model_pixel as mp
import model_mc as mc
import transform as t

def load_options(name):
    """Saves experiment options under names to load in train and test."""
    if name == 'pixel2d_atlas':
        idx = d.read_idx('../data/Template_files/'+
                         'Schaefer2018_400Parcels_7Networks_PCAordering.txt')
        tr = t.Transforms((t.Reorder(0, slice(1, None, None)), 
                           t.Reorder(0, idx),
                           t.ChannelDim()))
        # (img, lbl) is (1 x 1 x 400 x 200, 1 x 1 x 400 x 200) after batch
        train = d.RestAtlasDataset('../data/train', tr)
        test = d.RestAtlasDataset('../data/test', tr)
        model = mp.PixelCNN
        optimizer = optim.Adam
    if name == 'pixel2d_deep':
        idx = d.read_idx('../data/Template_files/'+
                         'Schaefer2018_400Parcels_7Networks_PCAordering.txt')
        tr = t.Transforms((t.Reorder(0, slice(1, None, None)), 
                           t.Reorder(0, idx),
                           t.ChannelDim()))
        train = d.RestAtlasDataset('../data/train', tr)
        test = d.RestAtlasDataset('../data/test', tr)
        model = lambda ch, shape: mp.PixelCNN(ch, shape, depth = 30)
        optimizer = optim.Adam
    if name == 'pixel2d_wide':
        idx = d.read_idx('../data/Template_files/'+
                         'Schaefer2018_400Parcels_7Networks_PCAordering.txt')
        tr = t.Transforms((t.Reorder(0, slice(1, None, None)), 
                           t.Reorder(0, idx),
                           t.ChannelDim()))
        train = d.RestAtlasDataset('../data/train', tr)
        test = d.RestAtlasDataset('../data/test', tr)
        model = lambda ch, shape: mp.PixelCNN(ch, shape, kernel = 5)
        optimizer = optim.Adam
    if name == 'markov_atlas':
        """
        RestAtlasDataset pre-transformation: 401 x 200 (atlas x time)
        post-transformation: 1 x 400 x 200
        Transforms:
        Reorder: gets rid of first atlas location (401 -> 400)
        NormalizeSum: treat blood as distribution w/ fixed sum
        """
        tr = t.Transforms((t.Reorder(0, slice(1, None, None)),
                           t.NormalizeSum(0),
                           t.ChannelDim()))
        train = d.RestAtlasDataset('../data/train', tr)
        test = d.RestAtlasDataset('../data/test', tr)
        model = mc.MarkovChain
        optimizer = optim.Adam
    example = train[0][0]
    ch, shape = example.shape[0], np.array(example.shape[1:])
    model = model(ch, shape)
    return {'dataset': (train, test), 'model': model, 'optimizer': optimizer}
