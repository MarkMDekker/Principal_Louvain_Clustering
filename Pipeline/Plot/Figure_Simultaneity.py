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
# ----------------------------------------------------------------- #

N = -1
SUB = '339295'
res = 200
tau = 30
regions = ['HIP', 'PAR', 'PFC']

# ----------------------------------------------------------------- #
# Start figure
# ----------------------------------------------------------------- #

fig = plt.figure(figsize=(20, 15))
gs = gridspec.GridSpec(3, 4)
gs.update(wspace=0.125, hspace=0.00)

# ----------------------------------------------------------------- #
# ----------------------------------------------------------------- #
# SIMULTANEITY PLOT
# ----------------------------------------------------------------- #
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
clus = list(np.unique(PAR)[:])+list(np.unique(PFC)[:])+list(np.unique(HIP)[:])
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
AdjMat = np.array(pd.read_pickle(pathact+'../Simultaneity/'+SUB+'.pkl'))
for i in tqdm(range(len(clus))):
    for j in range(i, len(clus)):
        if i != j:
            if AdjMat[i, j] > TH:
                edges.append((clus[i][:3]+' '+clus[i][4],
                              clus[j][:3]+' '+clus[j][4]))
                dicty[edges[-1]] = np.round(AdjMat[i, j], 2)
dictperc = {}
for i in range(len(clus)):
    s0, s1, s2, s3 = Expl(clus[i])
    dictperc[clus[i][:3]+' '+clus[i][4]] = [s1/AvMov, s2/AvExp]

axa = plt.subplot(gs[:, :3])
Network = nx.Graph()
Network.add_nodes_from(list(dictperc.keys()))
Network.add_edges_from(edges)
we = []
dicty2 = {}
for u,v in Network.edges:
    try:
        we.append(dicty[(u, v)])
    except:
        we.append(dicty[(v, u)])
    dicty2[(u, v)] = we[-1]
labels = {}
ns = []
cols2 = []
for node in Network.nodes():
    percs = dictperc[node]
    ns.append(percs[1])
    labels[node] = node+'\n'+str(np.round(percs[1],2))
    cols2.append(plt.cm.coolwarm((percs[1]-1)/1.5+0.5))
ns = np.array(ns)
we = np.array(we)
colors = ['forestgreen', 'tomato', 'steelblue', 'orange', 'violet',
          'brown']
cols = []
for i in Network.nodes:
    if i[:3] == 'PAR': cols.append(colors[0])
    if i[:3] == 'PFC': cols.append(colors[1])
    if i[:3] == 'HIP': cols.append(colors[2])
pos = nx.nx_agraph.graphviz_layout(Network, prog='neato')
nx.draw(Network, pos, node_size=2500, node_color=cols2,
        width=0.1+10*we, ax=axa)
axa.collections[0].set_edgecolor('k')
nx.draw_networkx_labels(Network, pos, labels, font_size=13, ax=axa)
nx.draw_networkx_edge_labels(Network, pos, edge_labels=dicty2, ax=axa)
lst = ['PAR', 'PFC', 'HIP']
# for i in range(len(lst)):
#     ax.text(0.02, 0.95-i*0.02, lst[i], color=colors[i],
#             fontsize=15, transform=ax.transAxes)
axa.text(0.02, 0.01, 'Threshold of '+str(TH), color='k',
         fontsize=15, transform=axa.transAxes, ha='left', va='bottom')
axa.text(0.02, 0.04, 'Rodent '+SUB, color='k',
         fontsize=15, transform=axa.transAxes, ha='left', va='bottom')

# ----------------------------------------------------------------- #
# ----------------------------------------------------------------- #
# PHASESPACE PLOTS
# ----------------------------------------------------------------- #
# ----------------------------------------------------------------- #

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
    UniClus = UniClus[UniClus != '-'    ]
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
    # Reshape to fit weights
    # --------------------------------------------------------------------- #

    Clus2d_v3 = np.copy(Clus2d_v2)
    uni = np.unique(Clus2d_v3[~np.isnan(Clus2d_v3)])
    for i in uni:
        Clus2d_v3[Clus2d_v3 == i] = dictperc[REG+' '+ss[int(i)]][1]

    # --------------------------------------------------------------------- #
    # Plot
    # --------------------------------------------------------------------- #

    ss = np.array(['A','B','C','D','E','F','G','H','I','J','K','L','M','N'])
    nclus = len(np.unique(Clus2d[~np.isnan(Clus2d)]))
    #ax.pcolormesh(Xvec, Yvec, Clus2d, cmap=plt.cm.Blues, vmin=-0.2, vmax=15)
    ax.pcolormesh(Xvec, Yvec, Clus2d_v3, cmap=plt.cm.coolwarm, vmin=0, vmax=2)#, edgecolors='k', linewidths=0.1)
    #ax.pcolormesh(Xvec, Yvec, Clus2d_inv, cmap=plt.cm.Greys, vmin=0, vmax=1)
    for i in range(int(np.nanmax(Clus2d)+1)):
        whX, whY = np.where(Clus2d == i)
        ax.text(np.median(Xvec[whY]), np.median(Yvec[whX]), ss[i], fontsize=11,
                bbox=dict(facecolor='w', alpha=0.6, edgecolor='k', lw=0.5,
                          boxstyle='round'), ha='center', va='center')
    if PLOT == 2:
        ax.set_xlabel('Principal Component 1', fontsize=15)
    if PLOT == 1:
        ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    if PLOT == 2:
        ax.tick_params(axis="x", direction="in", pad=-15)
    ax.tick_params(axis="y", direction="in", pad=-20)
    ax.set_ylim([-12, 12])
    ax.set_xlim([-12, 12])
    ax.text(0.5, 0.95, REG, fontsize=13, transform=ax.transAxes, va='top',
            ha='center')
fig.tight_layout()
#plt.savefig(Path_Figsave+'Rodent'+str(SUB)+'.png', bbox_inches='tight', dpi=200)
