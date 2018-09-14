import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt

from os.path import join

subj = '/home/steven/MRI/function/data/train/NC00Test/NC01'
img_atlas = nib.load(join(subj, 'rest_atlas_tproj.nii.gz')).get_data()
img = nib.load(join(subj, 'rest_postproc.nii.gz')).get_data()