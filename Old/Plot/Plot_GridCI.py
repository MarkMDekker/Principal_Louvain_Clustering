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

Str = 'PAR'
path_figsave = path_figsave + Str
N = 100000
res = 200
tau = 30
clus2ds = []
Exiss = []
for prt in range(5):
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
# Clusters of general graph
# --------------------------------------------------------------------- #

N_gen = 600000
prt_gen = 0
clus2ds_gen = []
Exiss_gen = []
s_gen = Str+'/'+str(N_gen)+'_'+str(prt_gen)
Dat = pd.read_csv(path_pdata+'Clusters/'+s_gen+'.csv')
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
space = np.linspace(np.floor(np.min([Xpc, Ypc])),
                    np.floor(np.max([Xpc, Ypc]))+1, res)
H, xedges, yedges = np.histogram2d(Xpc, Ypc, bins=[space, space])
xedges2 = (xedges[:-1] + xedges[1:])/2.
yedges2 = (yedges[:-1] + yedges[1:])/2.
Xvec = xedges2
Yvec = yedges2
res2 = len(xedges2)
Clus2d_gen = func12(ExisC).T
clus2ds_gen.append(Clus2d_gen)
Exiss_gen.append(Exis)


#%%
# --------------------------------------------------------------------- #
# Find overlap
# --------------------------------------------------------------------- #


def func12(vec):
    mat = np.zeros(shape=(res2, res2))
    a = 0
    for i in range(res2):
        #for j in range(res2):
        mat[i] = vec[a*res2:a*res2+res2]
        a += 1
    return mat


def overlap_metric(C1, C2):
    exis1 = np.copy(C1)
    exis2 = np.copy(C2)
    exis1[np.isnan(exis1)] = -1
    exis2[np.isnan(exis2)] = -1
    exis1_2d = gaussian_filter(func12(exis1), 1)
    exis2_2d = gaussian_filter(func12(exis2), 1)
    exis1 = func21(exis1_2d)
    exis2 = func21(exis2_2d)
    #exis1 = gaussian_filter(exis1, 1)
    #exis2 = gaussian_filter(exis2, 1)
    deel1 = np.where(exis1 > 0.3)[0]
    deel2 = np.where(exis2 > 0.3)[0]
    same = np.intersect1d(deel1, deel2)
    #dif1 = deel1[~np.isin(deel1, same)]
    #dif2 = deel2[~np.isin(deel2, same)]
    score = 2*len(same) / (1e-9+len(deel1) + len(deel2))
    return score
#%%
totn = 0
exis_flat = []
setlist = []
for i in range(len(Exiss)):
    totn += len(Exiss[i])
    for j in range(len(Exiss[i])):
        exis_flat.append(Exiss[i][j])
        setlist.append(i)
matrix = np.zeros(shape=(totn, totn))
for i in tqdm(range(totn)):
    for j in range(totn):
        if setlist[j] != setlist[i]:
            matrix[i, j] = overlap_metric(exis_flat[i], exis_flat[j])
#%%
# --------------------------------------------------------------------- #
# Network format
# --------------------------------------------------------------------- #

tuples = np.array(np.where((matrix > 0.4) & (matrix != 1))).T
colors = ['forestgreen', 'tomato', 'steelblue', 'orange', 'violet',
          'chocolate', 'gray', 'firebrick', 'dodgerblue', 'silver',
          'gold', 'b', 'r', 'm', '0.96', '0.7', '0.4', 'lightblue', 'purple',
          'yellow']
Network = nx.Graph()
Network.add_edges_from(tuples)
cols = []
for i in Network.nodes:
    cols.append(colors[setlist[i]])
edgeval = []
for u, v in Network.edges:
    edgeval.append(matrix[u, v])
edgeval = np.array(edgeval)
edgeval = (edgeval - np.min(edgeval)) / (np.max(edgeval) - np.min(edgeval))
pos = pygraphviz_layout(Network)
fig, ax = plt.subplots(figsize=(15, 15))
nx.draw(Network, pos, node_size=200, node_color = cols, width=0.1+2*edgeval)
nx.draw_networkx_labels(Network, pos, font_size=8)
ax.text(0.0, 0.95+1*0.03, 'Sets of '+str(int(N/1000))+' sec:', color='k',
        transform=ax.transAxes)
for i in range(len(np.unique(setlist))):
    ax.text(0.02, 0.95-i*0.02, 'Set '+str(i), color=colors[i],
            transform=ax.transAxes)
plt.savefig(path_figsave+'/Network_'+str(N)+'.png', bbox_inches='tight', dpi=200)

#%%
# --------------------------------------------------------------------- #
# Network with clustering
# --------------------------------------------------------------------- #

th = 0.4
tuples = np.array(np.where((matrix > th) & (matrix != 1))).T
colors = ['forestgreen', 'tomato', 'steelblue', 'orange', 'violet',
          'chocolate', 'gray', 'firebrick', 'dodgerblue', 'silver',
          'gold', 'b', 'r', 'm', '0.96', '0.7', '0.4', 'lightblue', 'purple',
          'yellow']
Network = nx.Graph()
Network.add_edges_from(tuples)
partition = community_louvain.best_partition(Network, weight='weight')
nums = {}
for i in Network.nodes:
    nums[i] = setlist[i]
cols = []
for i in Network.nodes:
    cols.append(colors[partition[i]])
edgeval = []
for u, v in Network.edges:
    edgeval.append(matrix[u, v])
edgeval = np.array(edgeval)
edgeval = (edgeval - np.min(edgeval)) / (np.max(edgeval) - np.min(edgeval))
pos = pygraphviz_layout(Network)
fig, ax = plt.subplots(figsize=(15, 15))
nx.draw(Network, pos, node_size=120, node_color = cols, width=0.1+2*edgeval)
nx.draw_networkx_labels(Network, pos, labels=nums, font_size=7)
plt.savefig(path_figsave+'/Network_clusters_'+str(N)+'_'+str(th)+'.png', bbox_inches='tight', dpi=200)

#%%
# --------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------- #

keys = np.array(list(partition.keys()))
vals = np.array(list(partition.values()))
Clusclus = []
for i in range(np.max(vals)+1):
    ke = keys[np.where(vals == i)[0]]
    if len(ke) >= 3:
        Clusclus.append(keys[np.where(vals == i)[0]])
strings = []

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 8),
                                             sharey=True, sharex=True)
axes = [ax1, ax2, ax3, ax4, ax5, ax6]
for a in range(6):
    ax = axes[a]
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    try:
        clus = Clusclus[a]
        for i in range(len(clus2ds)):
            ax.contourf(Xvec, Yvec, clus2ds[i], colors='lightgrey', zorder=-1)
        for j in range(len(clus)):
            strings.append('Set '+str(setlist[clus[j]]))
            ClusI = func12(exis_flat[clus[j]]).T
            ClusI[np.isnan(ClusI)] = -1
            ClusI = gaussian_filter(ClusI, 2)
            mask = np.zeros(ClusI.shape, dtype=bool)
            mask[ClusI < 0] = 1
            Clus2d_m = np.ma.array(ClusI, mask=mask)
            ax.contour(Xvec, Yvec, mask, [0.01], colors=colors[setlist[clus[j]]],
                       linewidths=2, zorder=1e99)
        for j in range(len(Exiss_gen[0])):
            ClusI = func12(Exiss_gen[0][j]).T
            ClusI[np.isnan(ClusI)] = -1
            ClusI = gaussian_filter(ClusI, 1)
            mask = np.zeros(ClusI.shape, dtype=bool)
            mask[ClusI < 0] = 1
            Clus2d_m = np.ma.array(ClusI, mask=mask)
            ax.contour(Xvec, Yvec, mask, [0.01], colors='k',
                       linewidths=1)
    except:
        continue
for i in range(len(np.unique(setlist))):
    ax2.text(0.02, 0.95-i*0.05, 'Set '+str(i), color=colors[i],
            transform=ax2.transAxes, weight='bold')

ax4.set_xlabel('Principal Component 1', fontsize=15)
ax5.set_xlabel('Principal Component 1', fontsize=15)
ax6.set_xlabel('Principal Component 1', fontsize=15)
ax4.set_ylabel('Principal Component 2', fontsize=15)
ax1.set_ylabel('Principal Component 2', fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.tick_params(axis='both', which='minor', labelsize=14)

ax.set_ylim([-6, 6])
ax.set_xlim([-6, 6])
fig.tight_layout()
plt.savefig(path_figsave+'/GroupedClusters_'+str(N)+'.png', bbox_inches='tight', dpi=200)

#%%
# --------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------- #

strings = ['(a) Average co-location: ',
           '(b) Average co-location: ',
           '(c) Average co-location: ',
           '(d) Average co-location: ',
           '(e) Average co-location: ',
           '(f) Average co-location: ']
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 8),
                                             sharey=True, sharex=True)
axes = [ax1, ax2, ax3, ax4, ax5, ax6]
for a in range(6):
    ax = axes[a]
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    try:
        clus = Clusclus[a]
        for i in range(len(clus2ds)):
            ax.contourf(Xvec, Yvec, clus2ds[i], colors='lightgrey', zorder=-1)
        totclus = np.zeros(shape=(res2, res2))
        for j in range(len(clus)):
            ClusI = func12(exis_flat[clus[j]]).T
            ClusI[np.isnan(ClusI)] = 0
            totclus += ClusI
        totclus = gaussian_filter(totclus, 1) / len(clus)
        mask = np.zeros(totclus.shape, dtype=bool)
        mask[totclus < 0.01] = 1
        totclus_m = np.ma.array(totclus, mask=mask)
        ax.contourf(Xvec, Yvec, totclus_m, np.arange(0.25, 1.1, 0.25),
                    cmap=plt.cm.Blues)
        ax.contour(Xvec, Yvec, totclus, np.arange(0.25, 1.1, 0.25), colors='k',
                   zorder=1e9)
        for j in range(len(Exiss_gen[0])):
            ClusI = func12(Exiss_gen[0][j]).T
            ClusI[np.isnan(ClusI)] = -1
            ClusI = gaussian_filter(ClusI, 1)
            mask = np.zeros(ClusI.shape, dtype=bool)
            mask[ClusI < 0] = 1
            Clus2d_m = np.ma.array(ClusI, mask=mask)
            ax.contour(Xvec, Yvec, mask, [0.01], colors='purple',
                       linewidths=2, zorder=1e9-1)
        ax.text(0.02, 0.98,
                strings[a]+str(np.round(np.mean(totclus[totclus > 0.25]), 3))+' ('+str(len(clus))+')',
                color='k',
                transform=ax.transAxes, fontsize=11, va='top')
    except:
        continue

ax4.set_xlabel('Principal Component 1', fontsize=15)
ax5.set_xlabel('Principal Component 1', fontsize=15)
ax6.set_xlabel('Principal Component 1', fontsize=15)
ax4.set_ylabel('Principal Component 2', fontsize=15)
ax1.set_ylabel('Principal Component 2', fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.tick_params(axis='both', which='minor', labelsize=14)

ax.set_ylim([-6, 6])
ax.set_xlim([-6, 6])
fig.tight_layout()

axc = fig.add_axes([0.1, -0.035, 0.8, 0.02])
norm = mpl.colors.Normalize(vmin=0, vmax=1)
cb1 = mpl.colorbar.ColorbarBase(axc, cmap=plt.cm.Blues, norm=norm,
                                orientation='horizontal')
cb1.set_label('Co-location of clusters in sub-time series', fontsize=15)
cb1.set_ticks(np.arange(0.0, 1.1, 0.25))
axc.tick_params(labelsize=13)

plt.savefig(path_figsave+'/ClusterDensity_'+str(N)+'.png', bbox_inches='tight', dpi=200)
