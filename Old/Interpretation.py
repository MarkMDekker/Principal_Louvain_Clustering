# ----------------------------------------------------------------- #
# About script
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
path_clus = ('/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/'
             'Output/ActiveClusters/F_')

# ----------------------------------------------------------------- #
# Input
# ----------------------------------------------------------------- #

# 339295
# 279418
NUM = '339295'

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
clus = np.array(clus)[np.array(clus) != '-']
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
    
#%%
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

# ----------------------------------------------------------------- #
# Plot graph
# ----------------------------------------------------------------- #

Network = nx.Graph()
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
fig, ax = plt.subplots(figsize=(15, 15))
nx.draw(Network, pos, node_size=3000, node_color=cols2,
        width=0.1+10*we)
ax = plt.gca()
ax.collections[0].set_edgecolor('k') 
nx.draw_networkx_labels(Network, pos, labels, font_size=14)
nx.draw_networkx_edge_labels(Network, pos, edge_labels=dicty2)
lst = ['PAR', 'PFC', 'HIP']
# for i in range(len(lst)):
#     ax.text(0.02, 0.95-i*0.02, lst[i], color=colors[i],
#             fontsize=15, transform=ax.transAxes)
ax.text(0.98, 0.02, 'Threshold of '+str(TH), color='k',
        fontsize=15, transform=ax.transAxes, ha='right', va='bottom')
ax.text(0.02, 0.02, 'Rodent '+NUM, color='k',
        fontsize=15, transform=ax.transAxes, ha='left', va='bottom')
#plt.savefig(Path_Figsave+'Simultaneity/'+NUM+'_v2.png', bbox_inches='tight', dpi=200)