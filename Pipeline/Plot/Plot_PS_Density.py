# ----------------------------------------------------------------- #
# About script
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import configparser
import networkx as nx
import matplotlib as mpl
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec
from community import community_louvain
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import sys
sys.path.insert(1, '../')
from Functions import func12, func21

# ----------------------------------------------------------------- #
# Input
# ----------------------------------------------------------------- #

#Subjects: 279418, 279419, 339295

N = 100000
REG = 'PFC'
SUB = '339295'
res = 200
tau = 30
th = 0.5
plot_network = 1
if N == -1:
    Name = REG+'_'+SUB
else:
    Name = REG+'_'+SUB+'_'+str(N)

# ----------------------------------------------------------------- #
# Read data
# ----------------------------------------------------------------- #

res2 = res-1
config = configparser.ConfigParser()
config.read('../config.ini')
Path_Data = config['PATHS']['DATA_RAW']
Path_Datasave = config['PATHS']['DATA_SAVE']
Path_Figsave = config['PATHS']['FIG_SAVE']
ClustersH = []
for j in range(6):
    ClustersH.append(np.array(pd.read_pickle(Path_Datasave+'Clusters/'+Name +
                                            '_'+str(j)+'.pkl')))
ClustersH = np.array(ClustersH)
MasterClustersH = np.array(pd.read_pickle(Path_Datasave+'Clusters/'+REG+'_' +
                                          SUB+'.pkl'))

# --------------------------------------------------------------------- #
# Relate to master clusters
# --------------------------------------------------------------------- #


def func12(vec):
    mat = np.zeros(shape=(res2, res2))
    a = 0
    for i in range(res2):
        mat[i] = vec[a*res2:a*res2+res2]
        a += 1
    return mat


def func21(mat):
    return mat.flatten()


def overlap_metric(Cm, Cc):
    exism = np.copy(Cm)
    exisc = np.copy(Cc)
    intersect = np.intersect1d(exisc, exism)
    score = len(intersect) / (1e-9+len(exisc))    
    return score

setlist = []
Clusters = []
for i in range(6):
    for j in range(int(np.nanmax(ClustersH[i].T[1])+1)):
        Clusters.append(np.where(ClustersH[i].T[1] == j)[0])
        setlist.append(i)

MasterClusters = []
for j in range(int(np.nanmax(MasterClustersH.T[1])+1)):
    MasterClusters.append(np.where(MasterClustersH.T[1] == j)[0])


SimMat = np.zeros(shape=(len(MasterClusters), len(Clusters)))
for i in range(len(MasterClusters)):
    for j in range(len(Clusters)):
        SimMat[i, j] = overlap_metric(MasterClusters[i], Clusters[j])

# --------------------------------------------------------------------- #
# Create network object
# --------------------------------------------------------------------- #

ss = np.array(['A','B','C','D','E','F','G','H','I','J','K','L','M','N'])
colors = ['forestgreen', 'tomato', 'steelblue', 'orange', 'violet' , 'brown']
edgesraw = np.array(np.where(SimMat > th))
edges = np.array([ss[edgesraw[0]], edgesraw[1]]).T
weights = SimMat[np.where(SimMat > th)]
dicty = {}
for i in range(len(edges)):
    dicty[tuple(edges[i])] = np.round(weights[i], 2)
Network = nx.Graph()
Network.add_edges_from(edges)
cols = []
for i in Network.nodes:
    try:
        cols.append(colors[setlist[int(i)]])
    except:
        cols.append('silver')
pos = nx.nx_agraph.graphviz_layout(Network, prog='neato')
if plot_network == 1:
    fig, ax = plt.subplots(figsize=(15, 15))
    nx.draw(Network, pos, node_size=1000, node_color=cols,
            width=0.1+5*weights**2)
    nx.draw_networkx_labels(Network, pos, font_size=15)
    nx.draw_networkx_edge_labels(Network, pos, edge_labels=dicty)
    ax.text(0.0, 0.95+1*0.03, 'Sets of '+str(int(N/1000))+' sec:', color='k',
            transform=ax.transAxes, fontsize=15)
    for i in np.unique(setlist):
        ax.text(0.02, 0.95-i*0.02, 'Set '+str(i), color=colors[i], fontsize=15,
                transform=ax.transAxes)
    ax.text(0.98, 0.02, 'Threshold of '+str(th), color='k', fontsize=15,
            transform=ax.transAxes, ha='right', va='bottom')
#%%
# --------------------------------------------------------------------- #
# Apply Louvain clustering
# --------------------------------------------------------------------- #

partition = community_louvain.best_partition(Network, weight='weight')
keys = np.array(list(partition.keys()))
vals = np.array(list(partition.values()))
weight_clus = []
Clusclus = []
for i in range(np.max(vals)+1):
    ke = keys[np.where(vals == i)[0]]
    if len(ke) >= 3:
        Clusclus.append(keys[np.where(vals == i)[0]])
        char = keys[np.where(vals == i)[0]][0]
        weight_clus.append(weights[np.where(edges.T[0]==char)])
#%%
# --------------------------------------------------------------------- #
# Create density fields
# --------------------------------------------------------------------- #

clus2ds = []
Exiss = []
for prt in range(len(ClustersH)):

    Dat = ClustersH[prt].T[1]
    nclus = len(np.unique(Dat[~np.isnan(Dat)]))
    Exis = np.zeros(shape=(nclus, len(Dat)))+np.nan
    ExisC = np.zeros(shape=(len(Dat)))+np.nan
    for i in range(nclus):
        Exis[i, Dat == i] = 1
        ExisC[Dat == i] = i
    res2 = res - 1
    Clus2d = func12(ExisC).T
    clus2ds.append(Clus2d)
    Exiss.append(Exis)

# --------------------------------------------------------------------- #
# Master clusters
# --------------------------------------------------------------------- #

Dat = MasterClustersH.T[1]
nclus = len(np.unique(Dat[~np.isnan(Dat)]))
MasterExis = np.zeros(shape=(nclus, len(Dat)))+np.nan
ExisC = np.zeros(shape=(len(Dat)))+np.nan
for i in range(nclus):
    MasterExis[i, Dat == i] = 1
    ExisC[Dat == i] = i
res2 = res - 1

# --------------------------------------------------------------------- #
# Get space
# --------------------------------------------------------------------- #

PCs = np.array(pd.read_pickle(Path_Datasave+'PCs/'+REG+'_'+SUB+'.pkl'))
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

Clus2d_gen = func12(ExisC).T
#clus2ds_gen.append(Clus2d_gen)
#Exiss_gen.append(Exis)

totn = 0
exis_flat = []
setlist = []
for i in range(len(Exiss)):
    totn += len(Exiss[i])
    for j in range(len(Exiss[i])):
        exis_flat.append(Exiss[i][j])
        setlist.append(i)
#%%

strings = ['(a) Average similarity: ',
           '(b) Average similarity: ',
           '(c) Average similarity: ',
           '(d) Average similarity: ',
           '(e) Average similarity: ',
           '(f) Average similarity: ',
           '(g) Average similarity: ',
           '(h) Average similarity: ',
           '(i) Average similarity: ']
#fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 8),
#                                             sharey=True, sharex=True)
cmap = plt.cm.magma_r
fig = plt.figure(figsize=(16, 8))
gs = gridspec.GridSpec(2, 4, wspace=0.02, hspace=0.02)
axes = []
for i in range(2):
    for j in range(4):
        axes.append(plt.subplot(gs[i, j]))
axes = np.array(axes)
for a in range(8):
    ax = axes[a]
    try:
        master = Clusclus[a][0]
        clus = [int(s) for s in Clusclus[a][1:] if s.isdigit()]
        #ax.contourf(Xvec, Yvec, func12(MasterExis[j]).T, colors='lightgrey', zorder=-1)
        for i in range(len(MasterExis)):
            ax.contourf(Xvec, Yvec, func12(MasterExis[i]).T, colors='lightgrey', zorder=-1)
        totclus = np.zeros(shape=(res2, res2))
        for j in range(len(clus)):
            ClusI = func12(exis_flat[clus[j]]).T
            ClusI[np.isnan(ClusI)] = 0
            totclus += ClusI
            #ClusI2 = func12(exis_flat[clus[j]]).T
            #ClusI2[np.isnan(ClusI2)] = -1
            #ClusI2 = gaussian_filter(ClusI2, 0.5)
            #mask = np.zeros(ClusI2.shape, dtype=bool)
            #mask[ClusI2 < 0] = 1
            #ax.contour(Xvec, Yvec, mask, [0.01], colors='k',
            #           linewidths=0.5, zorder=1e9)
        totclus = gaussian_filter(totclus, 1) / len(clus)
        mask = np.zeros(totclus.shape, dtype=bool)
        mask[totclus < 0.01] = 1
        totclus_m = np.ma.array(totclus, mask=mask)
        ax.contourf(Xvec, Yvec, totclus_m, np.arange(0.1, 1.1, 0.05),
                    cmap=cmap)
        #ax.contour(Xvec, Yvec, totclus, np.arange(0.1, 1.1, 0.1), colors='k',
        #           zorder=1e9, linewidths=0)
        for j in range(len(MasterExis)):
            ClusI = func12(MasterExis[j]).T
            ClusI[np.isnan(ClusI)] = -1
            ClusI = gaussian_filter(ClusI, 1)
            mask = np.zeros(ClusI.shape, dtype=bool)
            mask[ClusI < 0] = 1
            Clus2d_m = np.ma.array(ClusI, mask=mask)
            if ss[j] == master:
                ax.contour(Xvec, Yvec, mask, [0.01], colors='dodgerblue',
                           linewidths=2, zorder=1e9+2)
            else:
                ax.contour(Xvec, Yvec, mask, [0.01], colors='grey',
                           linewidths=1, zorder=1e9+1)
        ax.text(0.02, 0.98,
                strings[a]+str(np.round(np.mean(weight_clus[a]), 3))+' ('+str(len(clus))+')',
                color='k',
                transform=ax.transAxes, fontsize=11, va='top')
    except:
        pass

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim([-8, 8])
    ax.set_xlim([-8, 8])

    if a >= 4:
        ax.set_xlabel('PC1', fontsize=15)
        ax.set_xticks([-6, -3, 0, 3, 6])
    if a == 0 or a == 4:
        ax.set_ylabel('PC2', fontsize=15)
        ax.set_yticks([-6, -3, 0, 3, 6])


axc = fig.add_axes([0.12, 0.015, 0.78, 0.02])
norm = mpl.colors.Normalize(vmin=0, vmax=1)
cb1 = mpl.colorbar.ColorbarBase(axc, cmap=cmap, norm=norm,
                                orientation='horizontal')
cb1.set_label('Overlap of sub-clusters related to (blue) master cluster', fontsize=15)
cb1.set_ticks(np.arange(0.0, 1.1, 0.25))
axc.tick_params(labelsize=13)

plt.savefig(Path_Figsave+'/Dens/'+str(REG)+'_'+str(SUB)+'.png', bbox_inches='tight', dpi=200)
