import nibabel as nib
import numpy as np

import random
from fnmatch import fnmatch
from os import listdir
from os.path import join, isdir

class SubjectDataset:
    """Loads data in the following form:
    SubjectDataset(train, t, r):
    train >
        patient1 >
            ...
        patient2
        ...
    """
    def __init__(self, folder, transform, read):
        self.folder = folder
        self.subjects = [join(folder, s) for s in listdir(folder) if 
                         isdir(join(folder, s))]
        self.transform = transform
        self.read = read
        
    def __len__(self):
        return len(self.subjects)
    
    def __getitem__(self, i):
        return self.load(self.subjects[i])
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
        
    def load(self, filename):
        example = self.read(filename) # tuple (image, label)
        example = self.transform(example)
        return example
    
    def shuffle(self):
        random.shuffle(self.subjects)
        
class Split:
    """Abstract class defining the split operation on a dataset.
    As an example, you might want to split 3d data by slice.
    Faster if you get every split of 1 example before moving on (otherwise
    you end up reading each example from disk multiple times)
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.i = 0
        self.example = dataset[self.i]
        self.depth = None # depth = splits/example, depends on splitting
        
    def __len__(self):
        return len(self.dataset) * self.depth
    
    def __getitem__(self, i):
        i, d = i // self.depth, i % self.depth
        if i != self.i:
            self.i = i
            self.example = self.dataset[self.i]
        return self.pick(d)
    
    def __pick__(self, d):
        x, y = self.example
        s = self.slice_ind(d)
        return (x[s], y[s])
    
    def slice_ind(self, d):
        raise NotImplementedError
        
    def shuffle(self):
        self.dataset.shuffle()
        
class Split4th(Split):
    """Splits 4d arrays by 4th spatial dimension."""
    def __init__(self, dataset):
        super().__init__(dataset)
        self.depth = self.example[0].shape[4]
    
    def slice_ind(self, d):
        return np.index_exp[:,:,:,:,d]
    
class Split3rd(Split):
    """Splits 3d arrays by 3rd spatial dimension."""
    def __init__(self, dataset):
        super().__init__(dataset)
        self.depth = self.example[0].shape[3]
    
    def slice_ind(self, d):
        return np.index_exp[:,:,:,d]
    
class SplitPatch(Split):
    """Splits 3d arrays into non-overlapping patches."""
    def __init__(self, dataset, patches = 8):
        super().__init__(dataset)
        self.patches = patches
        self.depth = patches ** 3
        self.size = (np.array(self.example[0].shape[1:]) // patches)
        
    def slice_ind(self, d):
        """d is converted to a length 3 base (patches) number.
        Each digit is the starting point of the patch in that dimension.
        """
        d1, d = d % self.patches, d // self.patches
        d2, d3 = d % self.patches, d // self.patches
        d1 *= self.size[0]
        d2 *= self.size[1]
        d3 *= self.size[2]
        return np.index_exp[:,d1:d1+self.size[0],
                            d2:d2+self.size[1],
                            d3:d3+self.size[2]]

def read_image_label(r1, r2):
    def read(folder):
        return r1(folder), r2(folder)
    return read

def read_file(match):
    def f(folder):
        images = [join(folder, f) for f in listdir(folder) if
                  fnmatch(f, match)]
        image = nib.load(images[0]).get_data()
        return image
    return f

def DTIRestDataset(folder, transform):
    return SubjectDataset(folder, transform, 
                read_image_label(read_file('*tensor.nii.gz'), 
                                 read_file('*mcf_brain.nii.gz')))

def T2AutoencDataset(folder, transform):
    return SubjectDataset(folder, transform, 
                read_image_label(read_file('*T2*brain.nii.gz'), 
                                 read_file('*T2*brain.nii.gz')))