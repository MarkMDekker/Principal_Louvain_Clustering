# --------------------------------------------------------------------- #
# Preambule
# --------------------------------------------------------------------- #

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import warnings
import matplotlib as mpl
from Pygraphviz import pygraphviz_layout
from community import community_louvain
import networkx as nx
from scipy.ndimage.filters import gaussian_filter
warnings.filterwarnings("ignore")
path_pdata = ('/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/'
              'ProcessedData/')
path_figsave = ('/Users/mmdekker/Documents/Werk/Figures/Figures_Side/Brain/'
                'Pics_newest/')

# --------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------- #

#Clusclus = [[3, 12, 22, 43, 51, 60, 84, 91],
#            [72, 37, 24, 61, 83, 2, 11, 42, 92, 35, 50],
#            [85, 94, 6, 33, 52, 63, 13, 26, 46],
#            [7, 46, 73, 27, 16, 94]]
#
## 5 - PFC
#Clusclus = [[26, 1, 17, 9, 34],
#            [11, 27, 20, 35],
#            [7, 0, 32, 24, 16],
#            [22, 4, 13, 29, 37]]

Str = 'HIP'
path_figsave = path_figsave + Str
N = 600000
prt = 0
res = 200
tau = 30
clus2ds = []
Exiss = []
s = Str+'/'+str(N)+'_'+str(prt)

# --------------------------------------------------------------------- #
# Read data
# --------------------------------------------------------------------- #

Dat = pd.read_csv(path_pdata+'Clusters/'+s+'.csv')
nclus = len(np.unique(Dat.Cluster[~np.isnan(Dat.Cluster)]))
Exis = np.zeros(shape=(nclus, len(Dat)))+np.nan
ExisC = np.zeros(shape=(len(Dat)))+np.nan
for i in range(nclus):
    Exis[i, Dat.Cluster == i] = 1
    ExisC[Dat.Cluster == i] = i
res2 = res - 1
PCs = np.array(pd.read_pickle(path_pdata+'PCs/'+s+'.pkl'))
Xpc = PCs[0]
Ypc = PCs[1]
tau = 30

# --------------------------------------------------------------------- #
# Retrieve phase-space, calculate locations and rates
# --------------------------------------------------------------------- #

space = np.linspace(np.floor(np.min([Xpc, Ypc])),
                    np.floor(np.max([Xpc, Ypc]))+1, res)
H, xedges, yedges = np.histogram2d(Xpc, Ypc, bins=[space, space])
xedges2 = (xedges[:-1] + xedges[1:])/2.
yedges2 = (yedges[:-1] + yedges[1:])/2.
Xvec = xedges2
Yvec = yedges2
res2 = len(xedges2)

def func12(vec):
    mat = np.zeros(shape=(res2, res2))
    a = 0
    for i in range(res2):
        for j in range(res2):
            mat[i, j] = vec[a]
            a += 1
    return mat


def func21(mat):
    return mat.flatten()

# --------------------------------------------------------------------- #
# Create 2D and mask
# --------------------------------------------------------------------- #

Clus2d = func12(ExisC).T
clus2ds.append(Clus2d)
Exiss.append(Exis)

# --------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------- #

colors = ['forestgreen', 'tomato', 'steelblue', 'orange', 'violet',
          'chocolate', 'gray', 'firebrick', 'dodgerblue', 'silver',
          'gold', 'b', 'r', 'm','k','k','k','k','k']
fig, ax = plt.subplots(figsize=(8, 8))
for j in range(len(clus2ds)):
    Exis = Exiss[j]
    Clus2d = np.copy(clus2ds[j])
    nclus = len(np.unique(Clus2d[~np.isnan(Clus2d)]))
    Clus2d[np.isnan(Clus2d)] = -1
    Clus2d = gaussian_filter(Clus2d, 2)
    for i in range(nclus):
        ClusI = func12(Exis[i]).T
        ClusI = gaussian_filter(ClusI, 0.)
        ClusI[np.isnan(ClusI)] = -1
        mask = np.zeros(ClusI.shape, dtype=bool)
        mask[ClusI == -1] = 1
        Clus2d_m = np.ma.array(ClusI, mask=mask)
        plt.contour(Xvec, Yvec, mask, [0.01], colors='k', linewidths=2)

ax.tick_params(axis='both', which='major', labelsize=12)
ax.tick_params(axis='both', which='minor', labelsize=12)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.tick_params(axis='both', which='minor', labelsize=14)

ax.set_ylim([-7, 7])
ax.set_xlim([-7, 7])
fig.tight_layout()
plt.savefig(path_figsave+'/GenClusters.png', bbox_inches='tight', dpi=200)
