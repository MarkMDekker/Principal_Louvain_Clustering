# ----------------------------------------------------------------- #
# About script
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import pandas as pd
import numpy as np
import networkx as nx
from sklearn import metrics
from community import community_louvain
from sklearn.cluster import KMeans
from community import modularity
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
path = '/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/Output/'
path_clus = ('/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/'
             'Output/ActiveClusters/F_')


def windowmean(Data, size):
    ''' Function to compute a running window along a time series '''

    if size == 1:
        return Data
    elif size == 0:
        print('Size 0 not possible!')
        sys.exit()
    else:
        Result = np.zeros(len(Data)) + np.nan
        for i in range(np.int(size/2.), np.int(len(Data) - size/2.)):
            Result[i] = np.nanmean(Data[i-np.int(size/2.):i+np.int(size/2.)])
        return np.array(Result)


# ----------------------------------------------------------------- #
# Input
# ----------------------------------------------------------------- #

REG = 'HIP'
SUB = '346110'
RES = 200
space = np.linspace(-12, 12, RES)
#MAX = 1000000
#W = 25

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
Series = np.array([PC1_hip, PC2_hip, PC1_pfc, PC2_pfc, PC1_par, PC2_par])

#%%
# ----------------------------------------------------------------- #
# Check clustering 
# ----------------------------------------------------------------- #

S = 0
Series2 = Series[:, :40000:]
tau = 1
ks = np.arange(2, 300, 2)

mets = []
probss = []
for K in tqdm(ks):
    kmeans = KMeans(random_state=0, n_clusters=K).fit(Series2.T)
    labels = kmeans.labels_
    #inertia = kmeans.inertia_
    m = metrics.silhouette_score(Series2.T, labels, metric='euclidean')
    mets.append(m)
    probs = np.zeros(shape=(K, K))
    for k in range(K):
        wh = np.where(labels[:-tau] == k)[0]
        labs0 = labels[wh]
        labs1 = labels[wh+tau]
        tot = len(labs0)
        for i in range(K):
            probs[k, i] = len(labs1[labs1 == i])/tot
    probss.append(np.mean(np.diag(probs)))
    #centers = kmeans.cluster_centers

#%%
# ----------------------------------------------------------------- #
# Check invariance
# ----------------------------------------------------------------- #

probs = np.zeros(shape=(K, K))
for k in range(K):
    wh = np.where(labels[:-tau] == k)[0]
    labs0 = labels[wh]
    labs1 = labels[wh+tau]
    tot = len(labs0)
    for i in range(K):
        probs[k, i] = len(labs1[labs1 == i])/tot
#%%
# ----------------------------------------------------------------- #
# Plot invariance
# ----------------------------------------------------------------- #

plt.imshow(probs, cmap=plt.cm.gist_heat_r)
plt.colorbar()
#%%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
ax1.plot(ks, probss*(ks-1)/ks, 'o')
ax1.plot(ks, probss*(ks-1)/ks, c='steelblue')
ax1.set_ylabel('Average invariance (weighted)')
ax2.plot(ks, mets,'o', c='tomato')
ax2.plot(ks, mets, c='tomato')
ax2.set_ylabel('Silhouette coefficient')
ax2.set_xlabel('Amount of clusters (K)')

#%%
# ----------------------------------------------------------------- #
# Determine average
# ----------------------------------------------------------------- #

Loc_hip = np.array(pd.read_pickle(path+'Locations/F_HIP_'+SUB+'.pkl')).T[0]
Loc_pfc = np.array(pd.read_pickle(path+'Locations/F_PFC_'+SUB+'.pkl')).T[0]
Loc_par = np.array(pd.read_pickle(path+'Locations/F_PAR_'+SUB+'.pkl')).T[0]

uni_hip = np.unique(Loc_hip)
uni_pfc = np.unique(Loc_pfc)
uni_par = np.unique(Loc_par)

lab_hip = np.zeros(len(np.unique(Loc_hip)))
lab_pfc = np.zeros(len(np.unique(Loc_pfc)))
lab_par = np.zeros(len(np.unique(Loc_par)))

for l in tqdm(range(len(uni_hip))):
    loc = uni_hip[l]
    wh = np.where(Loc_hip == loc)[0]
    bestlab = np.argmax(np.bincount(labels[wh]))
    lab_hip[l] = bestlab

for l in tqdm(range(len(uni_pfc))):
    loc = uni_pfc[l]
    wh = np.where(Loc_pfc == loc)[0]
    bestlab = np.argmax(np.bincount(labels[wh]))
    lab_pfc[l] = bestlab

for l in tqdm(range(len(uni_par))):
    loc = uni_par[l]
    wh = np.where(Loc_par == loc)[0]
    bestlab = np.argmax(np.bincount(labels[wh]))
    lab_par[l] = bestlab

#%%
# ----------------------------------------------------------------- #
# Plot in phase-space
# ----------------------------------------------------------------- #

space = np.linspace(-12, 12, RES)
H, xedges, yedges = np.histogram2d([0], [0], bins=[space, space])
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


L_hip = np.zeros(len(func21(H)))+np.nan
L_hip[uni_hip] = lab_hip
L_hip = func12(L_hip)
L_pfc = np.zeros(len(func21(H)))+np.nan
L_pfc[uni_pfc] = lab_pfc
L_pfc = func12(L_pfc)
L_par = np.zeros(len(func21(H)))+np.nan
L_par[uni_par] = lab_par
L_par = func12(L_par)

#%%

regions = ['(a) HIP', '(b) PFC', '(c) PAR']
Ls = [L_hip, L_pfc, L_par]
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4), sharex=True,
                                    sharey=True)
axes = [ax1, ax2, ax3]
for i in range(3):
    ax = axes[i]
    L = Ls[i]
    ax.pcolormesh(xedges2, yedges2, L, cmap=plt.cm.rainbow)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    if i == 0:
        ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    ax.set_ylim([-12, 12])
    ax.set_xlim([-12, 12])
    ax.text(0.02, 0.98, regions[i], fontsize=16, transform=ax.transAxes,
            va='top')
fig.tight_layout()

#%%
# ----------------------------------------------------------------- #
# Plot in phase-space
# ----------------------------------------------------------------- #

N = 25000
PC1s = [PC1_hip, PC1_pfc, PC1_par]
PC2s = [PC2_hip, PC2_pfc, PC2_par]
regions = ['(a) HIP', '(b) PFC', '(c) PAR']
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4), sharex=True,
                                    sharey=True)
axes = [ax1, ax2, ax3]
for i in range(3):
    ax = axes[i]
    ax.scatter(PC1s[i][:N], PC2s[i][:N], c=labels[:N], s=3, cmap=plt.cm.rainbow)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    if i == 0:
        ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    ax.set_ylim([-12, 12])
    ax.set_xlim([-12, 12])
    ax.text(0.02, 0.98, regions[i], fontsize=16, transform=ax.transAxes,
            va='top')
fig.tight_layout()

#%%
# ----------------------------------------------------------------- #
# 3D plots
# ----------------------------------------------------------------- #

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d

R = 8
space = np.linspace(-R, R, RES)
N= 100000
C = 1
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.view_init(45, -45)
X, Y, Z = PC1_hip[:N], PC1_pfc[:N], PC1_par[:N]
ax.scatter(X, Y, Z, s=0.2, c='silver', alpha=0.01, zorder=0)
ax.scatter(X[labels[:N]==C], Y[labels[:N]==C], Z[labels[:N]==C], s=0.2,
           c='forestgreen', zorder=1e9)


Hz, xedges, yedges = np.histogram2d(X, Y, bins=[space, space])
Hx, xedges, yedges = np.histogram2d(Y, Z, bins=[space, space])
Hy, xedges, yedges = np.histogram2d(X, Z, bins=[space, space])
Hz[Hz == 0] = np.nan
Hx[Hx == 0] = np.nan
Hy[Hy == 0] = np.nan
xedges2 = (xedges[:-1] + xedges[1:])/2.
yedges2 = (yedges[:-1] + yedges[1:])/2.
xe, ye = np.meshgrid(xedges2, yedges2)
ax.contourf(xe, Hy.T, ye, 100, zdir='y', offset=R, cmap=plt.cm.gist_heat_r, zorder=-1)
ax.contourf(xe, ye, Hz.T, 100, zdir='z', offset=-R, cmap=plt.cm.gist_heat_r, zorder=-1)
ax.contourf(Hx.T, xe, ye, 100, zdir='x', offset=-R, cmap=plt.cm.gist_heat_r, zorder=-1)

wh = np.where(labels[:N] == C)[0]
Xi, Yi, Zi = PC1_hip[:N][wh], PC1_pfc[:N][wh], PC1_par[:N][wh]

Hz, xedges, yedges = np.histogram2d(Xi, Yi, bins=[space, space])
Hx, xedges, yedges = np.histogram2d(Yi, Zi, bins=[space, space])
Hy, xedges, yedges = np.histogram2d(Xi, Zi, bins=[space, space])
Hz[Hz == 0] = np.nan
Hx[Hx == 0] = np.nan
Hy[Hy == 0] = np.nan
xedges2 = (xedges[:-1] + xedges[1:])/2.
yedges2 = (yedges[:-1] + yedges[1:])/2.
xe, ye = np.meshgrid(xedges2, yedges2)
col = plt.cm.rainbow(C/np.max(labels))
ax.contourf(xe, Hy.T, ye, 100, zdir='y', offset=R, c=col, zorder=-1)
ax.contourf(xe, ye, Hz.T, 100, zdir='z', offset=-R, c=col, zorder=-1)
ax.contourf(Hx.T, xe, ye, 100, zdir='x', offset=-R, c=col, zorder=-1)

ax.set_xlabel('HIP')
ax.set_xlim(-R, R)
ax.set_ylabel('PFC')
ax.set_ylim(-R, R)
ax.set_zlabel('PAR')
ax.set_zlim(-R, R)
fig.tight_layout()
plt.show()
