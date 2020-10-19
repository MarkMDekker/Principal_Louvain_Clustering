# ----------------------------------------------------------------- #
# About script
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import pandas as pd
import numpy as np
import networkx as nx
from community import community_louvain
from community import modularity
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
path = '/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/Output/'
path_clus = ('/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/'
             'Output/ActiveClusters/F_')

# ----------------------------------------------------------------- #
# Input
# ----------------------------------------------------------------- #

REG = 'HIP'
SUB = '346110'
RES = 200
space = np.linspace(-12, 12, RES)

# ----------------------------------------------------------------- #
# Read time series
# ----------------------------------------------------------------- #

PCs_hip = np.array(pd.read_pickle(path+'PCs/F_HIP_'+SUB+'.pkl'))[:2]
PCs_par = np.array(pd.read_pickle(path+'PCs/F_PAR_'+SUB+'.pkl'))[:2]
PCs_pfc = np.array(pd.read_pickle(path+'PCs/F_PFC_'+SUB+'.pkl'))[:2]

PC1_hip = PCs_hip[0]
PC2_hip = PCs_hip[1]
PC1_pfc = PCs_pfc[0]
PC2_pfc = PCs_pfc[1]
PC1_par = PCs_par[0]
PC2_par = PCs_par[1]
xpcs = [PC1_hip, PC1_pfc, PC1_par]
ypcs = [PC2_hip, PC2_pfc, PC2_par]

# ----------------------------------------------------------------- #
# Get locations and clusters
# ----------------------------------------------------------------- #

Loc_hip = np.array(pd.read_pickle(path+'Locations/F_HIP_'+SUB+'.pkl')).T[0]
Loc_pfc = np.array(pd.read_pickle(path+'Locations/F_PFC_'+SUB+'.pkl')).T[0]
Loc_par = np.array(pd.read_pickle(path+'Locations/F_PAR_'+SUB+'.pkl')).T[0]

Clus_hip = np.array(pd.read_pickle(path+'ActiveClusters/F_HIP_'+SUB+'.pkl').HIP)
Clus_pfc = np.array(pd.read_pickle(path+'ActiveClusters/F_PFC_'+SUB+'.pkl').PFC)
Clus_par = np.array(pd.read_pickle(path+'ActiveClusters/F_PAR_'+SUB+'.pkl').PAR)

# ----------------------------------------------------------------- #
# Get average positions
# ----------------------------------------------------------------- #

H, xedges, yedges = np.histogram2d(xpcs[0], ypcs[0], bins=[space, space])
xedges2 = (xedges[:-1] + xedges[1:])/2.
yedges2 = (yedges[:-1] + yedges[1:])/2.
RES2 = len(xedges2)
xe = np.array([xedges2]*len(ypcs[0]))
ye = np.array([yedges2]*len(xpcs[0]))

# ----------------------------------------------------------------- #
# Get Clusters in space
# ----------------------------------------------------------------- #


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
Hs = []
a = 0
xvecs = []
yvecs = []
for REG in tqdm(['HIP', 'PAR', 'PFC']):
    mina = np.min([xpcs[a], ypcs[a]])
    maxa = np.max([xpcs[a], ypcs[a]])
    space = np.linspace(-12, 12, RES)
    H, xedges, yedges = np.histogram2d(xpcs[a], ypcs[a], bins=[space, space])
    xedges2 = (xedges[:-1] + xedges[1:])/2.
    yedges2 = (yedges[:-1] + yedges[1:])/2.
    RES2 = len(xedges2)
    xe = np.array([xedges2]*len(ypcs[a]))
    ye = np.array([yedges2]*len(xpcs[a]))
    ss = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                   'L', 'M', 'N'])
    ClusDF = pd.read_pickle(path+'Clusters/F_'+REG+'_'+SUB+'.pkl')
    ClusAct = pd.read_pickle(path+'ActiveClusters/F_'+REG+'_'+SUB+'.pkl')
    UniClus = np.unique(ClusAct[REG])
    UniClus = UniClus[UniClus != '-']
    UniClusN = []
    for i in range(len(UniClus)):
        UniClusN.append(np.where(ss == UniClus[i])[0][0])
    UniClusN = np.array(UniClusN)
    nclus = len(np.unique(ClusDF.Cluster[~np.isnan(ClusDF.Cluster)]))
    Exis = np.zeros(shape=(nclus, len(ClusDF)))+np.nan
    ExisC = np.zeros(shape=(len(ClusDF)))+np.nan
    for i in range(nclus):
        Exis[i, ClusDF.Cluster == i] = 1
        ExisC[ClusDF.Cluster == i] = i
    res2 = RES - 1
    Clus2d = func12(ExisC)
    Clus2d_inv = np.copy(Clus2d)
    Clus2d_inv[np.isnan(Clus2d)] = 0
    Clus2d_inv[~np.isnan(Clus2d)] = np.nan
    Clus2d_filt = gaussian_filter(Clus2d_inv, 0.125)
    Clus2d_v2 = np.copy(Clus2d)
    for i in range(len(Clus2d_v2)):
        for j in range(len(Clus2d_v2)):
            if Clus2d_v2[i, j] not in UniClusN:
                Clus2d_v2[i, j] = np.nan
    Xvec = xedges2
    Yvec = yedges2
    xvecs.append(Xvec)
    yvecs.append(Yvec)
    Hs.append(Clus2d_v2)
    a += 1

unilochip = np.unique(Loc_hip) 
unilochip2 = unilochip[::5]
unilochipc = np.unique(Loc_hip[Clus_hip != '-'])
X = []
Y = []
U_par = []
U_pfc = []
V_par = []
V_pfc = []
conv = np.zeros(shape=(199, 199))

for i in tqdm(range(len(unilochipc))):
    loc = unilochipc[i]
    wh = np.where(Loc_hip == loc)[0]

    hip1 = PC1_hip[wh]
    hip2 = PC2_hip[wh]
    pfc1 = PC1_pfc[wh]
    pfc2 = PC2_pfc[wh]
    par1 = PC1_par[wh]
    par2 = PC2_par[wh]

    avpar1 = np.mean(par1)
    avpar2 = np.mean(par2)
    avpfc1 = np.mean(pfc1)
    avpfc2 = np.mean(pfc2)
    avhip1 = np.mean(hip1)
    avhip2 = np.mean(hip2)

    if len(wh) > -1:
        U_par.append(avpar1-avhip1)
        U_pfc.append(avpfc1-avhip1)
        V_par.append(avpar2-avhip2)
        V_pfc.append(avpfc2-avhip2)
        X.append(avhip1)
        Y.append(avhip2)
    else:
        continue
X = np.array(X)
Y = np.array(Y)
U_par = np.array(U_par)
V_par = np.array(V_par)

unilochipc = np.unique(Loc_hip[Clus_hip != '-'])
SimMat = np.zeros(shape=(len(unilochipc), len(unilochipc)))
AdjMat = np.zeros(shape=(len(unilochipc), len(unilochipc)))

for i in tqdm(range(len(SimMat))):
    X1 = U_par[i]+X[i]
    Y1 = V_par[i]+Y[i]
    X2s = U_par+X
    Y2s = V_par+Y
    dis = 1/(1+np.sqrt((X1-X2s)**2 + (Y1-Y2s)**2))
    dis[i] = 1
    SimMat[i] = dis

    dis2 = np.sqrt((X[i]-X)**2 + (Y[i]-Y)**2)
    AdjMat[i, dis2 < 0.35] = 1
CombMat = np.multiply(AdjMat, SimMat)

#%%

''' Generate network object '''
G = nx.from_numpy_array(CombMat)

''' Apply clustering algorithm '''
partition = community_louvain.best_partition(G, weight='weight')
Modularity = modularity(partition, G, weight='weight')

#%%
''' Obtain clusters '''
count = 0.
lens = []
clusters = []
for com in set(partition.values()):
    count = count+1.
    list_nodes = [nodes for nodes in partition.keys()
                  if partition[nodes] == com]
    clusters.append(list_nodes)
    lens.append(len(list_nodes))

#%%
DF = {}
DF['Cell'] = np.arange(len(func21(H)))
clus = np.zeros(len(DF['Cell']))+np.nan
for i in range(len(clusters)):
    for j in clusters[i]:
        clus[unilochipc[j]] = int(i)
DF['Cluster'] = clus
DF = pd.DataFrame(DF)
#DF.to_pickle('/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/Output/ClustersNew/HIP.pkl')

#%%
# ----------------------------------------------------------------- #
# Plot
# ----------------------------------------------------------------- #


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
Hs = []
a = 0
xvecs = []
yvecs = []

ClusDF = DF
nclus = len(np.unique(ClusDF.Cluster[~np.isnan(ClusDF.Cluster)]))
UniClusN = range(nclus)
Exis = np.zeros(shape=(nclus, len(ClusDF)))+np.nan
ExisC = np.zeros(shape=(len(ClusDF)))+np.nan
for i in range(nclus):
    Exis[i, ClusDF.Cluster == i] = 1
    ExisC[ClusDF.Cluster == i] = i
res2 = RES - 1
Clus2d = func12(ExisC)
Clus2d_inv = np.copy(Clus2d)
Clus2d_inv[np.isnan(Clus2d)] = 0
Clus2d_inv[~np.isnan(Clus2d)] = np.nan
Clus2d_filt = gaussian_filter(Clus2d_inv, 0.125)
Clus2d_v2 = np.copy(Clus2d)
for i in range(len(Clus2d_v2)):
    for j in range(len(Clus2d_v2)):
        if Clus2d_v2[i, j] not in UniClusN:
            Clus2d_v2[i, j] = np.nan
Xvec = xedges2
Yvec = yedges2

#%%
# ----------------------------------------------------------------- #
# Recompute arrows
# ----------------------------------------------------------------- #

unilochip = np.unique(Loc_hip) 
CLUS = 1
clustercells = np.unique(DF.Cell[DF.Cluster == CLUS])
X = []
Y = []
U_par = []
U_pfc = []
V_par = []
V_pfc = []
conv = np.zeros(shape=(199, 199))

for i in tqdm(range(len(clustercells))):
    loc = clustercells[i]
    wh = np.where(Loc_hip == loc)[0]

    hip1 = PC1_hip[wh]
    hip2 = PC2_hip[wh]
    pfc1 = PC1_pfc[wh]
    pfc2 = PC2_pfc[wh]
    par1 = PC1_par[wh]
    par2 = PC2_par[wh]

    avpar1 = np.mean(par1)
    avpar2 = np.mean(par2)
    avpfc1 = np.mean(pfc1)
    avpfc2 = np.mean(pfc2)
    avhip1 = np.mean(hip1)
    avhip2 = np.mean(hip2)

    if len(wh) > -1:
        U_par.append(avpar1-avhip1)
        U_pfc.append(avpfc1-avhip1)
        V_par.append(avpar2-avhip2)
        V_pfc.append(avpfc2-avhip2)
        X.append(avhip1)
        Y.append(avhip2)
    else:
        continue

# ----------------------------------------------------------------- #
# Create histogram
# ----------------------------------------------------------------- #

Us = [U_par, U_pfc]
Vs = [V_par, V_pfc]
st = ['(a) Projecting HIP to PAR', '(b) Projecting HIP to PFC']
var = [[PC1_par, PC2_par], [PC1_pfc, PC2_pfc]]
fig, ax = plt.subplots(1, 1, figsize=(5, 5), sharey=True)
for i in range(1):
    #ax = axes[i]
    mina = np.min([xpcs[i], ypcs[i]])
    maxa = np.max([xpcs[i], ypcs[i]])
    space = np.linspace(-12, 12, RES)
    hist, x, y = np.histogram2d(var[i][0], var[i][1], bins=[space, space],
                                density=True)
    hist = gaussian_filter(hist, 1)
    ax.quiver(X, Y, Us[i], Vs[i], angles='xy', scale_units='xy', scale=1,
              width=0.001, zorder=1e9, alpha=0.25)
    ax.pcolormesh(xedges2, yedges2, Clus2d_v2, vmin=-np.nanmax(Clus2d_v2)*0.2, vmax=np.nanmax(Clus2d_v2)*1.2,
                  cmap=plt.cm.Greys)
    ax.scatter(X, Y, c='tomato', s=5, alpha=0.15, zorder=1e99)
    ax.text(0.02, 0.98, st[i], fontsize=12, ha='left', va='top',
            transform=ax.transAxes)
    ax.text(0.02, 0.93, 'Modularity '+str(np.round(Modularity,2)),
            fontsize=10, ha='left', va='top',
            transform=ax.transAxes)
    ax.text(0.02, 0.88, str(nclus)+' clusters',
            fontsize=10, ha='left', va='top',
            transform=ax.transAxes)
    ax.set_xlim([-12, 12])
    ax.set_ylim([-12, 12])
    ax.set_xlabel('Principal Component 1', fontsize=11)
#axes[0].set_ylabel('Principal Component 2', fontsize=11)
ax.set_ylabel('Principal Component 2', fontsize=11)
fig.tight_layout()

# plt.savefig('/Users/mmdekker/Documents/Werk/Figures/Figures_Side/Brain/'
#             'Pipeline/Quiver_Simultaneity2/'+SUB+'_'+str(CLUS)+'.png',
#             bbox_inches='tight', dpi=200)

