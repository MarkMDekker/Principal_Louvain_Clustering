# ----------------------------------------------------------------- #
# About script
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import configparser
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
pathact = ('/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/Output'
           '/ActiveClusters/')

# ----------------------------------------------------------------- #
# Input
# ----------------------------------------------------------------- #

#Subjects: 279418, 279419, 339295

N = -1
SUB = '279418'
res = 200
tau = 30

regions = ['HIP', 'PAR', 'PFC']
fig = plt.figure(figsize=(21, 10))
gs = gridspec.GridSpec(3, 4)
gs.update(wspace=0.025, hspace=0.00)
axa = plt.subplot(gs[:, :3])
for PLOT in range(3):
    ax = plt.subplot(gs[PLOT, 3])
    REG = regions[PLOT]
    if N == -1:
        Name = 'F_'+REG+'_'+SUB
    else:
        Name = REG+'_'+SUB+'_'+str(N)
    
    # ----------------------------------------------------------------- #
    # Read data
    # ----------------------------------------------------------------- #
    
    ss = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                   'L', 'M', 'N'])
    config = configparser.ConfigParser()
    config.read('../config.ini')
    Path_Data = config['PATHS']['DATA_RAW']
    Path_Datasave = config['PATHS']['DATA_SAVE']
    Path_Figsave = config['PATHS']['FIG_SAVE']
    PCs = np.array(pd.read_pickle(Path_Datasave+'PCs/'+Name+'.pkl'))
    ClusDF = pd.read_pickle(Path_Datasave+'Clusters/'+Name+'.pkl')
    ClusAct = pd.read_pickle(pathact+Name+'.pkl')
    UniClus = np.unique(ClusAct[REG])#[1:]
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
    res2 = res - 1
    Xpc = PCs[0]
    Ypc = PCs[1]
    
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
    Clus2d_inv = np.copy(Clus2d)
    Clus2d_inv[np.isnan(Clus2d)] = 0
    Clus2d_inv[~np.isnan(Clus2d)] = np.nan
    Clus2d_filt = gaussian_filter(Clus2d_inv, 0.125)
    Clus2d_v2 = np.copy(Clus2d)
    for i in range(len(Clus2d_v2)):
        for j in range(len(Clus2d_v2)):
            if Clus2d_v2[i, j] not in UniClusN:
                Clus2d_v2[i, j] = np.nan
    
    # --------------------------------------------------------------------- #
    # Plot
    # --------------------------------------------------------------------- #
    
    ss = np.array(['A','B','C','D','E','F','G','H','I','J','K','L','M','N'])
    nclus = len(np.unique(Clus2d[~np.isnan(Clus2d)]))
    ax.pcolormesh(Xvec, Yvec, Clus2d, cmap=plt.cm.Blues, vmin=-0.2, vmax=15)
    ax.pcolormesh(Xvec, Yvec, Clus2d_v2, cmap=plt.cm.Reds, vmin=-2, vmax=15)#, edgecolors='k', linewidths=0.1)
    #ax.pcolormesh(Xvec, Yvec, Clus2d_inv, cmap=plt.cm.Greys, vmin=0, vmax=1)
    for i in range(int(np.nanmax(Clus2d)+1)):
        whX, whY = np.where(Clus2d == i)
        ax.text(np.median(Xvec[whY]), np.median(Yvec[whX]), ss[i],
                bbox=dict(facecolor='w', alpha=1, edgecolor='k', lw=2.5,
                          boxstyle='round'))
        ax.text(np.median(Xvec[whY]), np.median(Yvec[whX]), ss[i],
                bbox=dict(facecolor='w', alpha=1, edgecolor='k', lw=1.5,
                          boxstyle='round'))
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    if PLOT == 2:
        ax.set_xlabel('Principal Component 1', fontsize=15)
    if PLOT == 1:
        ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    ax.set_ylim([-12, 12])
    ax.set_xlim([-12, 12])
fig.tight_layout()
#plt.savefig(Path_Figsave+'Clus/'+Name+'.png', bbox_inches='tight', dpi=200)
