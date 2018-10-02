import numpy as np
#from sklearn.manifold import MDS
from sklearn.decomposition import PCA

file = ('/home/steven/MRI/function/data/Template_files/'+
        'Schaefer2018_400Parcels_7Networks_order.txt')
coords = open(file, 'r').readlines()
coords = [s.split()[2:5] for s in coords]
coords = np.array(coords).astype(int)

#embedding = MDS(n_components=1, metric=False)
#coords_1d_mds = np.squeeze(embedding.fit_transform(coords))
#idx_mds = np.argsort(coords_1d_mds)

# project onto first pca component?
proj = PCA(n_components = 1)
coords_1d_pca = np.squeeze(proj.fit_transform(coords))
idx_pca = np.argsort(coords_1d_pca)

def distances(idx):
    dist = np.zeros((len(idx), len(idx)))
    for i in range(len(idx)):
        for j in range(len(idx)):
            dist[i,j] = np.linalg.norm(idx[i] - idx[j])
    return dist

#dist_act, dist_mds, dist_pca = distances(coords), distances(idx_mds), distances(idx_pca)

dist_act, dist_pca = distances(coords), distances(coords_1d_pca)
