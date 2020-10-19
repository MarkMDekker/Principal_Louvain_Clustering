# ----------------------------------------------------------------- #
# About script
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

#%%

# ----------------------------------------------------------------- #
# Information on full phase-space
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
for REG in ['HIP', 'PAR', 'PFC']:
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
#%%
clusorpoint = 0

if clusorpoint == 0:
    # ----------------------------------------------------------------- #
    # Get locations grid cell and associated other regions
    # ----------------------------------------------------------------- #
    
    X = -3.5
    Y = -2
    XG = np.abs(xe.T-Y).argmin(axis=0)[0]
    YG = np.abs(ye.T-X).argmin(axis=0)[0]
    G = XG*len(xedges2)+YG
    wh = np.where(Loc_hip == G)[0]
else:
    # ----------------------------------------------------------------- #
    # Alternatively, use a cluster
    # ----------------------------------------------------------------- #

    CLUS = 'K'
    wh = np.where(Clus_hip == CLUS)[0]

pfc1 = PC1_pfc[wh]
pfc2 = PC2_pfc[wh]
par1 = PC1_par[wh]
par2 = PC2_par[wh]
hip1 = PC1_hip[wh]
hip2 = PC2_hip[wh]


# ----------------------------------------------------------------- #
# Create histogram
# ----------------------------------------------------------------- #

st = ['(a) HIP', '(b) PAR', '(c) PFC']
var = [[hip1, hip2], [par1, par2], [pfc1, pfc2]]
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for i in range(3):
    ax = axes[i]
    mina = np.min([xpcs[i], ypcs[i]])
    maxa = np.max([xpcs[i], ypcs[i]])
    space = np.linspace(-12, 12, RES)
    #x2 = Xvec[np.where(Hs[i] > -1)[0]]
    #y2 = Yvec[np.where(Hs[i] > -1)[1]]
    #space = np.linspace(np.min([x2, y2]), np.max([x2, y2]), 200)
    hist, x, y = np.histogram2d(var[i][0], var[i][1], bins=[space, space],
                                density=True)
    hist = gaussian_filter(hist, 1)

    ax.pcolormesh(xvecs[i], yvecs[i], Hs[i], vmin=-5, vmax=20, cmap=plt.cm.Greys)
    if i == 0 and clusorpoint == 0:
        ax.scatter([X], [Y], s=150, zorder=1e9, edgecolor='k', color='tomato')
    #else:
    ax.pcolormesh(xvecs[i], yvecs[i], 
                  np.ma.masked_where(hist < 0.0001, hist).T,
                  cmap=plt.cm.Spectral_r)
    ax.text(0.02, 0.98, st[i], fontsize=12, ha='left', va='top',
            transform=ax.transAxes)
    ax.set_xlim([-12, 12])
    ax.set_ylim([-12, 12])
    ax.set_xlabel('Principal Component 1', fontsize=11)
axes[0].set_ylabel('Principal Component 2', fontsize=11)
fig.tight_layout()
#plt.savefig('/Users/mmdekker/Documents/Werk/Figures/Figures_Side/Brain/'
#            'Pipeline/PDF_Simultaneity/'+SUB+'_'+CLUS+'.png',
#            bbox_inches='tight', dpi=200)

#%%

locs = np.unique(Loc_hip)
coords_hip = []
for l in range(len(locs)):
    loc = locs[l]
    YG = Yvec[int(np.mod(loc, len(xedges2)))]
    XG = Xvec[int(np.round((loc - YG) / len(xedges2)))]
    coords_hip.append((XG, YG))