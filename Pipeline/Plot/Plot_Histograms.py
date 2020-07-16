# ----------------------------------------------------------------- #
# About script
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import pandas as pd
import networkx as nx
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from tqdm import tqdm3
path_clus = ('/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/'
             'Output/ActiveClusters/F_')

# ----------------------------------------------------------------- #
# Input
# ----------------------------------------------------------------- #

# 339295
# 279418
# '279418T'
NUM = '279418'

# ----------------------------------------------------------------- #
# Read data
# ----------------------------------------------------------------- #

HIP = np.array(pd.read_pickle(path_clus+'HIP_'+NUM+'.pkl').HIP)
HIP = np.array(['HIP_'+i for i in HIP])
HIP[HIP == 'HIP_-'] = '-'
PFC = np.array(pd.read_pickle(path_clus+'PFC_'+NUM+'.pkl').PFC)
PFC = np.array(['PFC_'+i for i in PFC])
PFC[PFC == 'PFC_-'] = '-'
PAR = np.array(pd.read_pickle(path_clus+'PAR_'+NUM+'.pkl').PAR)
PAR = np.array(['PAR_'+i for i in PAR])
PAR[PAR == 'PAR_-'] = '-'
ACT = np.array(pd.read_pickle(path_clus+'PAR_'+NUM+'.pkl').Act)

# ----------------------------------------------------------------- #
# Calculations
# ----------------------------------------------------------------- #

def simultaneity(set1, set2, clus1, clus2):
    which1 = np.where(set1 == clus1)[0]
    which2 = np.where(set2 == clus2)[0]
    hit = len(np.intersect1d(which1, which2))
    #total = len(np.unique(list(which1) + list(which2)))
    totalav = np.min([len(np.unique(list(which1))), len(np.unique(list(which2)))])
    return hit / totalav

sets = [HIP, PFC, PAR]
clus = list(np.unique(PAR)[:])+list(np.unique(PFC)[:])+list(np.unique(HIP)[:])
sets = []
for i in range(len(clus)):
    if clus[i][:3] == 'HIP': sets.append(HIP)
    if clus[i][:3] == 'PFC': sets.append(PFC)
    if clus[i][:3] == 'PAR': sets.append(PAR)

# ----------------------------------------------------------------- #
# Determine exploration
# ----------------------------------------------------------------- #

AvMov = len(np.where(ACT == 1)[0])/len(ACT)
AvExp = len(np.where(ACT >= 2)[0])/len(ACT)


def Expl(clus):
    if clus[:3] == 'HIP': set1 = HIP
    if clus[:3] == 'PFC': set1 = PFC
    if clus[:3] == 'PAR': set1 = PAR
    which = np.where(set1 == clus)[0]
    s0 = len(np.where(ACT[which] == 0)[0])/len(which)
    s1 = len(np.where(ACT[which] == 1)[0])/len(which)
    s2 = len(np.where(ACT[which] >= 2)[0])/len(which)
    s3 = len(np.where(ACT[which] == 3)[0])/len(which)
    return s0, s1, s2, s3
    
# ----------------------------------------------------------------- #
# Edges and adjacency matrix
# ----------------------------------------------------------------- #

TH = 0.25
edges = []
dicty = {}
AdjMat = np.zeros(shape=(len(clus), len(clus)))
for i in tqdm(range(len(clus))):
    for j in range(i, len(clus)):
        if i != j:
            AdjMat[i, j] = simultaneity(sets[i], sets[j], clus[i], clus[j])
            AdjMat[j, i] = AdjMat[i, j]
            if AdjMat[i, j] > TH:
                edges.append((clus[i][:3]+' '+clus[i][4], clus[j][:3]+' '+clus[j][4]))
                dicty[edges[-1]] =  np.round(AdjMat[i, j], 2)

dictperc = {}
for i in range(len(clus)):
    s0, s1, s2, s3 = Expl(clus[i])
    dictperc[clus[i][:3]+' '+clus[i][4]] = [s1/AvMov, s2/AvExp]

#%%
# ----------------------------------------------------------------- #
# Create histogram from edge and node labels
# ----------------------------------------------------------------- #

nodelabels = np.array(list(dictperc.values())).T[1]
edgelabels = np.copy(AdjMat[AdjMat > 0])
mean = np.mean(edgelabels)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.hist(edgelabels, bins=np.linspace(0, 1, 50))
ax1.set_ylim([ax1.get_ylim()[0], ax1.get_ylim()[1]])
ax1.plot([mean, mean], [-1e3, 1e3], 'k', lw=2, zorder=1)
ax1.set_xlabel('Simultaneity (nonzero edge labels)'+'\n' +
               'Total: '+str(len(edgelabels)), fontsize=15)

ax2.hist(nodelabels, bins=np.linspace(0, 2, 25))
ax2.set_ylim([ax2.get_ylim()[0], ax2.get_ylim()[1]])
ax2.plot([1, 1], [-1e3, 1e3], 'k', lw=2, zorder=1)
ax2.set_xlabel('Exploration bias (node labels)'+'\n' +
               'Total: '+str(len(nodelabels)), fontsize=15)

fig.tight_layout()