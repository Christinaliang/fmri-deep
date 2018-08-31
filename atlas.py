import nibabel as nib
import numpy as np

import sys

def atlas_sum(atlas, img):
    """Takes in registered image and atlas and combines pixels in each atlas
    region. Both should be same resolution and shape.
    image should not be detrended! detrend after.
    """
    num_regions = np.max(atlas) + 1 # do this before masking!
    mask = img[:,:,:,0] != 0
    atlas *= mask
    summed = None
    for i in range(num_regions):
        time_series = img[np.where(atlas == i)] # pixels x time_pts
        sum_series = np.sum(time_series, axis = 0)
        if summed is None:
            summed = sum_series
        else:
            summed = np.vstack((summed, sum_series))
    return summed

def sum_to_volume(atlas, summed):
    """Takes in time series summed over atlas regions to create a volume
    where all pixels in a region have the same value (the summed value).
    """
    num_regions, time_pts = summed.shape
    img = np.zeros(atlas.shape + (time_pts,))
    assert num_regions == (np.max(atlas) + 1), ('Summed vector length should' + 
                           'be number of regions: Expected: ' + str(np.max(
                            atlas) + 1) + ', Actual: ' + str(num_regions))
    for i in range(num_regions):
        for t in range(time_pts):
            idx = np.where(atlas == i)
            idx = idx + tuple((np.zeros((1, idx[0].size)) + t).astype(int))
            img[idx] = summed[i, t] / max(1, idx[0].size)
    return img

def save_nii(name, img, affine = None):
    if affine is None:
        affine = np.eye(4)
    image = nib.Nifti1Image(img, affine)
    nib.save(image, name)

def main():
    if len(sys.argv) != 3:
        print('Arguments:')
        print('1: file path to atlas')
        print('2: file path to img')
        print('Example:')
        print('python atlas.py atlas.nii.gz img.nii.gz')
        return
    atlas = nib.load(sys.argv[1])
    affine = atlas.affine
    atlas = atlas.get_data()
    img = nib.load(sys.argv[2]).get_data()
    summed = atlas_sum(atlas, img)
    print(summed.shape)
    img_summed = sum_to_volume(atlas, summed)
    print(img_summed.shape)
    save_nii('rest_atlas_vec.nii.gz', summed)
    save_nii('rest_atlas_img.nii.gz', img_summed, affine = affine)
    
if __name__ == '__main__':
    main()