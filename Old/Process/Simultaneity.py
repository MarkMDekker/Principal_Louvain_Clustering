'''
Basic question: what clusters in what brain region light up and which coincide?
Most importantly, when the rodent is exploring: which combination of clusters
light up?

Steps:
    1. Find out simultaneous clusters in total set
    2. Find out simulatenous clusters in subset-aggregates
    3. Relate to exploration
'''

# --------------------------------------------------------------------------- #
# Preambule
# --------------------------------------------------------------------------- #

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Func_Locations import func_locations
from Pygraphviz import pygraphviz_layout
path_pdata = ('/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/'
              'ProcessedData/')

# --------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------- #

N = 600000
prt = 0

# --------------------------------------------------------------------- #
# Retrieve locations
# --------------------------------------------------------------------- #

Loc_hip = func_locations('HIP', N, prt, path_pdata)
Loc_pfc = func_locations('PFC', N, prt, path_pdata)
Loc_par = func_locations('PAR', N, prt, path_pdata)

#%%
# --------------------------------------------------------------------- #
# Read clusters
# --------------------------------------------------------------------- #

Clus_hip = np.array(pd.read_csv(path_pdata+'Clusters/HIP/600000_0.csv').Cluster)
Clus_pfc = np.array(pd.read_csv(path_pdata+'Clusters/PFC/600000_0.csv').Cluster)
Clus_par = np.array(pd.read_csv(path_pdata+'Clusters/PAR/600000_0.csv').Cluster)

#%%
# --------------------------------------------------------------------- #
# Determine clusters per datapoint
# --------------------------------------------------------------------- #

Time_clus = np.zeros(shape=(3, len(Loc_hip)))
for i in range(len(Loc_hip)):
    c0 = Clus_hip[Loc_hip[i]]
    c1 = Clus_pfc[Loc_pfc[i]]
    c2 = Clus_par[Loc_par[i]]
    Time_clus[0, i] = c0
    Time_clus[1, i] = c1
    Time_clus[2, i] = c2

# --------------------------------------------------------------------- #
# Simultaneity metric
# --------------------------------------------------------------------- #

AdjMat = np.zeros(shape=(8+5+6, 8+5+6))
for i in range(8):
    wh = np.where(Time_clus[0] == i)[0]
    uni_pfc = np.unique(Time_clus[1, wh]).astype(int)
    uni_par = np.unique(Time_clus[2, wh]).astype(int)
    uni_pfc = uni_pfc[uni_pfc > -1]
    uni_par = uni_par[uni_par > -1]
    for j in uni_pfc:
        wh2 = np.where(Time_clus[1, wh] == j)[0]
        AdjMat[i, j+8] = len(wh2) / len(wh)
    for j in uni_par:
        wh3 = np.where(Time_clus[2, wh] == j)[0]
        AdjMat[i, j+8+5] = len(wh3) / len(wh)

for i in range(5):
    wh = np.where(Time_clus[1] == i)[0]
    uni_hip = np.unique(Time_clus[0, wh]).astype(int)
    uni_par = np.unique(Time_clus[2, wh]).astype(int)
    uni_hip = uni_hip[uni_hip > -1]
    uni_par = uni_par[uni_par > -1]
    for j in uni_hip:
        wh2 = np.where(Time_clus[0, wh] == j)[0]
        AdjMat[i+8, j] = len(wh2) / len(wh)
    for j in uni_par:
        wh3 = np.where(Time_clus[2, wh] == j)[0]
        AdjMat[i+8, j+8+5] = len(wh3) / len(wh)

for i in range(6):
    wh = np.where(Time_clus[2] == i)[0]
    uni_hip = np.unique(Time_clus[0, wh]).astype(int)
    uni_pfc = np.unique(Time_clus[1, wh]).astype(int)
    uni_hip = uni_hip[uni_hip > -1]
    uni_pfc = uni_pfc[uni_pfc > -1]
    for j in uni_hip:
        wh2 = np.where(Time_clus[0, wh] == j)[0]
        AdjMat[i+8+5, j] = len(wh2) / len(wh)
    for j in uni_pfc:
        wh3 = np.where(Time_clus[1, wh] == j)[0]
        AdjMat[i+8+5, j+8] = len(wh3) / len(wh)

#%%
# --------------------------------------------------------------------- #
# Create network
# --------------------------------------------------------------------- #

tuples = np.array(np.where(AdjMat > 0.4)).T
colors = ['forestgreen', 'tomato', 'steelblue', 'orange', 'violet',
          'chocolate', 'gray', 'firebrick', 'dodgerblue', 'silver',
          'gold', 'b', 'r', 'm', '0.96', '0.7', '0.4', 'lightblue', 'purple',
          'yellow']
Network = nx.DiGraph()
Network.add_edges_from(tuples)
cols = []
for i in Network.nodes:
    if i < 8:
        cols.append('forestgreen')
    elif i < 8+5:
        cols.append('tomato')
    else:
        cols.append('steelblue')
edgeval = []
for u, v in Network.edges:
    edgeval.append(AdjMat[u, v])
edgeval = np.array(edgeval)
#edgeval = (edgeval - np.min(edgeval)) / (np.max(edgeval) - np.min(edgeval))
pos = pygraphviz_layout(Network)
fig, ax = plt.subplots(figsize=(15, 15))
nx.draw(Network, pos, node_size=500, node_color=cols)
nx.draw_networkx_edges(Network, pos, width=0.1+10*edgeval)
nx.draw_networkx_labels(Network, pos, font_size=10)
