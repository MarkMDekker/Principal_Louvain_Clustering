# ----------------------------------------------------------------- #
# About script
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import sys
from tqdm import tqdm
import networkx as nx
warnings.filterwarnings("ignore")

path_pc = ('/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/'
           'Output/PCs/')

# ----------------------------------------------------------------- #
# Input
# ----------------------------------------------------------------- #

SUBs = ['339295', '279418']
TH = 0.2
REGs = ['HIP', 'PAR', 'PFC']

# ----------------------------------------------------------------- #
# Calculations
# ----------------------------------------------------------------- #


def windowmean(Data, size):
    if size == 1:
        return Data
    elif size == 0:
        print('Size 0 not possible!')
        sys.exit()
    else:
        Result = np.zeros(len(Data))+np.nan
        for i in range(np.int(size/2.), np.int(len(Data)-size/2.)):
            Result[i] = np.nanmean(Data[i-np.int(size/2.):i+np.int(size/2.)])
        return np.array(Result)

Cs = []
for SUB in tqdm(SUBs):
    pcs_a = []
    pcs_e = []
    pcs_n = []
    regs = []
    for REG in REGs:
        Name = 'F_'+REG+'_'+SUB
        #Cat = pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/'
        #                     'SideProjects/Braindata/Output/Exploration/' +
        #                     Name+'.pkl')
        pcs_ = np.array(pd.read_pickle(path_pc+Name+'.pkl'))
        #Cat = np.ones(len(pcs_))
        #wh_nonexpl = np.where(Cat == 1)[0]
        #wh_expl = np.where(Cat > 1)[0]
        pc1 = pcs_[0]
        pc2 = pcs_[1]
        pc3 = pcs_[2]
        pcs_a = pcs_a + [pc1, pc2, pc3]
        #pcs_e = pcs_e + [pc1[wh_expl], pc2[wh_expl], pc3[wh_expl]]
        #pcs_n = pcs_n + [pc1[wh_nonexpl], pc2[wh_nonexpl], pc3[wh_nonexpl]]
        regs = regs + [REG, REG, REG]
    pcs_a = np.array(pcs_a)
    #cs_e = np.array(pcs_e)
    #pcs_n = np.array(pcs_n)
    Cs.append(np.corrcoef(pcs_a))


#%%
# ----------------------------------------------------------------- #
# Create graph object and plot
# ----------------------------------------------------------------- #

fig, axes = plt.subplots(1, 2, figsize=(15, 7))
for P in range(2):
    ax = axes[P]
    C = np.copy(Cs[P])
    edges = []
    dicty = {}
    for i in range(9):
        for j in range(9):
            if np.abs(C[i, j]) > TH and i != j:
                edges.append((i, j))
                dicty[(i, j)] = np.round(C[i, j], 2)
    Network = nx.Graph()
    Network.add_nodes_from(range(9))
    Network.add_edges_from(edges)
    
    we = []
    for u, v in Network.edges():
        we.append(np.abs(dicty[(u, v)]))
    we = np.array(we)
    
    labels = {}
    cols = []
    colors = ['steelblue', 'forestgreen', 'tomato']
    for node in Network.nodes():
        labels[node] = regs[node]+'\n'+'PC'+str(int(1+np.mod(node, 3)))
        if regs[node] == 'PAR': cols.append(colors[0])
        if regs[node] == 'PFC': cols.append(colors[1])
        if regs[node] == 'HIP': cols.append(colors[2])
    
    if P == 0:
        pos = nx.nx_agraph.graphviz_layout(Network, prog='neato')
    nx.draw(Network, pos, node_size=1500, node_color=cols, ax=ax, width=10*we**2)
    ax.collections[0].set_edgecolor('k')
    nx.draw_networkx_labels(Network, pos, labels, font_size=12, ax=ax)
    nx.draw_networkx_edge_labels(Network, pos, edge_labels=dicty, ax=ax)
    ax.set_xlim([ax.get_xlim()[0]-10, ax.get_xlim()[1]+10])
    ax.set_ylim([ax.get_ylim()[0]-10, ax.get_ylim()[1]+10])
    ax.text(0.5, 0.01, 'Rodent '+SUBs[P], color='k',
            fontsize=11, transform=ax.transAxes, ha='center', va='bottom')

plt.savefig('/Users/mmdekker/Documents/Werk/Figures/Figures_Side/Brain/'
            'Pipeline/EigVec/CorrelationsPCs.png', bbox_inches='tight', dpi=200)