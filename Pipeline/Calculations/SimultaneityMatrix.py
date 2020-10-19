# ----------------------------------------------------------------- #
# About script
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import configparser
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
pathact = ('/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/Output'
           '/ActiveClusters/')
path_clus = ('/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/'
             'Output/ActiveClusters/F_')


# ----------------------------------------------------------------- #
# Input
# 339295
# 279418
# ----------------------------------------------------------------- #

N = -1
SUB = '346110'
res = 200
tau = 30
regions = ['HIP', 'PAR', 'PFC']
CAT = 1

# ----------------------------------------------------------------- #
# Calculations
# ----------------------------------------------------------------- #

HIP = np.array(pd.read_pickle(path_clus+'HIP_'+SUB+'.pkl').HIP)
HIP = np.array(['HIP_'+i for i in HIP])
HIP[HIP == 'HIP_-'] = '-'
PFC = np.array(pd.read_pickle(path_clus+'PFC_'+SUB+'.pkl').PFC)
PFC = np.array(['PFC_'+i for i in PFC])
PFC[PFC == 'PFC_-'] = '-'
PAR = np.array(pd.read_pickle(path_clus+'PAR_'+SUB+'.pkl').PAR)
PAR = np.array(['PAR_'+i for i in PAR])
PAR[PAR == 'PAR_-'] = '-'
ACT = np.array(pd.read_pickle(path_clus+'PAR_'+SUB+'.pkl').Act)
clus = list(np.unique(PAR)[:])+list(np.unique(PFC)[:])+list(np.unique(HIP)[:])

if CAT == 1:
    HIP = HIP[ACT == 1]
    PFC = PFC[ACT == 1]
    PAR = PAR[ACT == 1]
if CAT == 2:
    HIP = HIP[ACT > 1]
    PFC = PFC[ACT > 1]
    PAR = PAR[ACT > 1]


def simultaneity(set1, set2, clus1, clus2):
    which1 = np.where(set1 == clus1)[0]
    which2 = np.where(set2 == clus2)[0]
    hit = len(np.intersect1d(which1, which2))
    totalav = np.min([len(np.unique(list(which1))),
                      len(np.unique(list(which2)))])
    return hit / totalav


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


sets = [HIP, PFC, PAR]
clus = np.array(clus)[np.array(clus) != '-']
sets = []
for i in range(len(clus)):
    if clus[i][:3] == 'HIP': sets.append(HIP)
    if clus[i][:3] == 'PFC': sets.append(PFC)
    if clus[i][:3] == 'PAR': sets.append(PAR)
AvMov = len(np.where(ACT == 1)[0])/len(ACT)
AvExp = len(np.where(ACT >= 2)[0])/len(ACT)
TH = 0.25
edges = []
dicty = {}
AdjMat = np.zeros(shape=(len(clus), len(clus)))
for i in tqdm(range(len(clus))):
    for j in range(i, len(clus)):
        if i != j:
            AdjMat[i, j] = simultaneity(sets[i], sets[j], clus[i], clus[j])
            AdjMat[j, i] = AdjMat[i, j]

# Save matrix
pd.DataFrame(AdjMat).to_pickle(pathact+'../Simultaneity/'+SUB+'_'+str(CAT)+'.pkl')
