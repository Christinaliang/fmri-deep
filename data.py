import nibabel as nib
import numpy as np

import random
from fnmatch import fnmatch
from os import listdir
from os.path import join, isdir

import transform as t

class Dataset:
    """Abstract class for loading examples from directory."""
    def __init__(self, folder, transform, read):
        self.folder = folder
        self.transform = transform
        self.read = read
        
    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, i):
        raise NotImplementedError
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
            
    def load(self, filename):
        example = self.read(filename)
        example = self.transform(example)
        image, label = example
        return (t.ToTensor()(image), t.ToTensor()(label))
    
    def shuffle(self):
        pass

class SubjectDataset(Dataset):
    """Loads data in the following form:
    SubjectDataset(train, t, r):
    train >
        patient1 >
            ...
        patient2
        ...
    """
    def __init__(self, folder, transform, read):
        super().__init__(folder, transform, read)
        self.subjects = [join(folder, s) for s in listdir(folder) if 
                         isdir(join(folder, s))]
        
    def __len__(self):
        return len(self.subjects)
    
    def __getitem__(self, i):
        # print(self.subjects[i])
        return self.load(self.subjects[i])
    
    def shuffle(self):
        random.shuffle(self.subjects)

def read_image_label(r1, r2 = None):
    if r2 == None:
        def read(folder): # image and label are the same
            r = r1(folder)
            return r, r
    else:
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

def read_idx(file):
    return np.squeeze(np.array(open(file, 'r').readlines()).astype(int))

def RestAtlasDataset(folder, transform):
    read = read_image_label(read_file('rest_atlas_vec400.nii.gz'))
    return SubjectDataset(folder, transform, read)