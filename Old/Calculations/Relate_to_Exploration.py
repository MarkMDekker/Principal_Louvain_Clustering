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

# ----------------------------------------------------------------- #
# Read time series
# ----------------------------------------------------------------- #

PCs_hip = np.array(pd.read_pickle(path+'PCs/F_HIP_'+SUB+'.pkl'))[:2]
PCs_par = np.array(pd.read_pickle(path+'PCs/F_PAR_'+SUB+'.pkl'))[:2]
PCs_pfc = np.array(pd.read_pickle(path+'PCs/F_PFC_'+SUB+'.pkl'))[:2]
EOFs_hip = np.array(pd.read_pickle(path+'EOFs/F_HIP_'+SUB+'.pkl'))
EOFs_par = np.array(pd.read_pickle(path+'EOFs/F_PAR_'+SUB+'.pkl'))
EOFs_pfc = np.array(pd.read_pickle(path+'EOFs/F_PFC_'+SUB+'.pkl'))
Res_hip = np.array(pd.read_pickle(path+'ResidualSeries/F_HIP_'+SUB+'.pkl'))
Res_par = np.array(pd.read_pickle(path+'ResidualSeries/F_PAR_'+SUB+'.pkl'))
Res_pfc = np.array(pd.read_pickle(path+'ResidualSeries/F_PFC_'+SUB+'.pkl'))

PC1_hip = PCs_hip[0]
PC2_hip = PCs_hip[1]
PC1_pfc = PCs_pfc[0]
PC2_pfc = PCs_pfc[1]
PC1_par = PCs_par[0]
PC2_par = PCs_par[1]
Series = np.array([PC1_hip, PC2_hip, PC1_pfc, PC2_pfc, PC1_par, PC2_par])
eof_series = np.array([EOFs_hip[0], EOFs_hip[1],
                       EOFs_pfc[0], EOFs_pfc[1],
                       EOFs_par[0], EOFs_par[1]])

#%%
# ----------------------------------------------------------------- #
# Tag data into clusters
# ----------------------------------------------------------------- #

Series2 = np.copy(Series)
for i in range(6):
    Series2[i][np.abs(Series2[i]) < 1.5] = 0

Clusters = (32*np.heaviside(Series2[0], 0.5) +
            16*np.heaviside(Series2[1], 0.5) +
            8*np.heaviside(Series2[2], 0.5) +
            4*np.heaviside(Series2[3], 0.5) +
            2*np.heaviside(Series2[4], 0.5) +
            np.heaviside(Series2[5], 0.5))

#%%
# ----------------------------------------------------------------- #
# Exploration data
# ----------------------------------------------------------------- #

Data = np.array(pd.read_pickle(path+'Exploration/F_HIP_'+SUB+'.pkl'))
nans = np.where(np.isnan(Data))[0]
wh = np.where(nans[1:]-nans[:-1] > 1)[0][0]
nan = nans[wh+1]
Data2 = np.array(list(Data[500:nan-500])+list(Data[500+nan:-500])).T[0]

# ----------------------------------------------------------------- #
# Biases
# ----------------------------------------------------------------- #

tots = []
biases = np.zeros(shape=(2, 64))
for i in range(64):
    wh = np.where(Clusters == i)[0]
    dat = Data2[wh]
    tot = len(dat[dat >= 1])
    if tot > 1000:
        c1 = len(np.where(dat == 1)[0])
        c2 = len(np.where(dat >= 2)[0])
        biases[0, i] = c1/tot
        biases[1, i] = c2/tot
    else:
        biases[0, i] = np.nan
        biases[1, i] = np.nan        
    tots.append(tot)

#%%
# ----------------------------------------------------------------- #
# Plot
# ----------------------------------------------------------------- #

cols = ['steelblue', 'tomato', 'forestgreen']
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
ax.plot(biases[0], c=cols[0], label='Not exploring')
ax.plot(biases[0], 'o', c=cols[0])
ax.plot(biases[1], c=cols[1], label='Exploring')
ax.plot(biases[1], 'o', c=cols[1])
ax2.set_xlabel('Index cluster')
ax.set_ylabel('Percentage')
ax2.plot(tots, 'ko')
ax2.plot(tots, 'k')
ax2.set_ylabel('Cluster size')
ax.legend(fontsize=9)
fig.tight_layout()

#%%
# ----------------------------------------------------------------- #
# EOFs
# ----------------------------------------------------------------- #

clus = np.where(biases[1] == np.nanmax(biases[1]))[0][0]
#%%
def func_sign(clus):
    return [np.heaviside(clus-32, 1),
            np.heaviside(np.mod(clus, 32)-16, 1),
            np.heaviside(np.mod(clus, 16)-8, 1),
            np.heaviside(np.mod(clus, 8)-4, 1),
            np.heaviside(np.mod(clus, 4)-2, 1),
            np.heaviside(np.mod(clus, 2)-1, 1)]

#%%
# ----------------------------------------------------------------- #
# EOFs plot
# ----------------------------------------------------------------- #

cols = ['steelblue', 'tomato', 'forestgreen']
nams = [r'$\delta$', r'$\theta$', r'$\beta$', r'$\gamma$']
cols2 = ['dodgerblue', 'forestgreen', 'tomato', 'goldenrod']
lows = [0, 5, 12, 32]
Res = [Res_hip, Res_pfc, Res_par]
upps = [5, 12, 32, 100]
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
ax1.text(0.02, 0.98, '(a) HIP', fontsize=16, va='top', transform=ax1.transAxes)
ax2.text(0.02, 0.98, '(b) PFC', fontsize=16, va='top', transform=ax2.transAxes)
ax3.text(0.02, 0.98, '(c) PAR', fontsize=16, va='top', transform=ax3.transAxes)

for k, ax in enumerate([ax1, ax2, ax3]):
    # for clus in cluss:
    #     signs = np.array(func_sign(clus))*2-1
    #     ax.plot(range(0, 100, 2), signs[2*k]*eof_series[2*k]+signs[2*k+1]*eof_series[2*k+1], 'k', lw=1)
    #     ax.plot(range(0, 100, 2), signs[2*k]*eof_series[2*k]+signs[2*k+1]*eof_series[2*k+1], 'ko')
    wh = np.where(Clusters == clus)[0]
    res = Res[k][:, wh]
    mean = np.mean(res, axis=1)
    std = np.std(res, axis=1)
    #for w in wh[:100]:
    ax.plot(range(0, 100, 2), mean, 'k', lw=2)
    ax.plot(range(0, 100, 2), mean+std, 'k', lw=0.5)
    ax.plot(range(0, 100, 2), mean-std, 'k', lw=0.5)
        #ax.plot(range(0, 100, 2), Res[k][:, w]], 'ko')
    ax.set_xlim([0, 100])
    ax.set_ylim([-2, 2])
    ax.plot([-1e3, 1e3], [0, 0], 'k', lw=0.5)
    for j in list(lows)+list(upps):
        ax.plot([j, j], [-1e3, 1e3], 'k', lw=0.5)
    for j in range(len(nams)):
        x = np.arange(lows[j], upps[j]+0.1, 1)
        y1 = [-1e3]*len(x)
        y2 = [1e3]*len(x)
        ax.fill_between(x, y1, y2, y2 >= y1, facecolor=cols2[j],
                        alpha=0.12,
                        interpolate=True, zorder=-1e10)
        ax.text(np.mean([lows[j], upps[j]]), ax.get_ylim()[0]+0.01,
                nams[j],
                va='bottom', ha='center', c=cols2[j], fontsize=15)
ax.set_xlabel('Frequency')
fig.tight_layout()