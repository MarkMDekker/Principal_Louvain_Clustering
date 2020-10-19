# ----------------------------------------------------------------- #
# About script
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import pandas as pd
import numpy as np
import community as community_louvain
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
path = '/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/Output/'

# ----------------------------------------------------------------- #
# Input
# ----------------------------------------------------------------- #

SUB = '346111'
RES = 10
TAU = 30
space = np.linspace(-12, 12, RES)

# ----------------------------------------------------------------- #
# Read time series
# ----------------------------------------------------------------- #

Data = np.array(pd.read_pickle(path+'Exploration/F_HIP_'+SUB+'.pkl'))
halfcut = len(np.array(pd.read_pickle(path+'Exploration/HIP_'+SUB+'.pkl')
                       ).T[0])
Data2 = np.array(list(Data[500:halfcut-500])+list(Data[500+halfcut:-500])).T[0]
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

#%% --------------------------------------------------------------- #
# Locations
# ----------------------------------------------------------------- #

Nclus = (RES-1)**6

def find_cluster(idx, space, Nclus):
    p1h = np.where(PC1_hip[idx] > space)[0][-1]
    p2h = np.where(PC2_hip[idx] > space)[0][-1]
    p1f = np.where(PC1_pfc[idx] > space)[0][-1]
    p2f = np.where(PC2_pfc[idx] > space)[0][-1]
    p1p = np.where(PC1_par[idx] > space)[0][-1]
    p2p = np.where(PC2_par[idx] > space)[0][-1]
    return str(p1h)+'-'+str(p2h)+'-'+str(p1f)+'-'+str(p2f)+'-'+str(p1p)+'-'+str(p2p)

Locations = []
for i in tqdm(range(len(PC1_hip))):
    Locations.append(find_cluster(i, space, Nclus))
Locations = np.array(Locations)
UniLocs = np.unique(Locations)

UniLocs_sep = []
for i in UniLocs:
    UniLocs_sep.append(np.array(i.split('-')).astype(int))
UniLocs_sep = np.array(UniLocs_sep)

#%% --------------------------------------------------------------- #
# Transfer Matrix
# ----------------------------------------------------------------- #

Mat = np.zeros(shape=(len(UniLocs), len(UniLocs)))
for i in tqdm(range(len(Mat))):
    wh = np.where(Locations == UniLocs[i])[0]
    for j in range(len(wh)):
        try:
            whj = np.where(UniLocs == Locations[wh[j]+TAU])[0][0]
            Mat[i, whj] += 1
        except:
            continue

for i in range(len(Mat)):
    Mat[i] = Mat[i]/np.sum(Mat[i])

#%% --------------------------------------------------------------- #
# Non-weighted Matrix
# ----------------------------------------------------------------- #

Mat2 = np.zeros(shape=(len(UniLocs), len(UniLocs)))
for i in tqdm(range(len(Mat))):
    for j in range(i, len(Mat)):
        Mat2[i, j] = np.mean([Mat[i, j], Mat[j, i]])
    
#%% --------------------------------------------------------------- #
# Clustering
# ----------------------------------------------------------------- #
    
G = nx.from_numpy_matrix(np.matrix(Mat2), create_using=nx.Graph)
partition = community_louvain.best_partition(G)
clusters = np.array(list(partition.values()))
modularity = community_louvain.modularity(partition, G)

#%% --------------------------------------------------------------- #
# Clusters in time
# ----------------------------------------------------------------- #

clusters_TS = []
for i in tqdm(range(len(Locations))):
    loc = Locations[i]
    wh = np.where(UniLocs == loc)[0][0]
    clusters_TS.append(clusters[wh])
clusters_TS = np.array(clusters_TS)

#%% --------------------------------------------------------------- #
# Relate to exploration data
# ----------------------------------------------------------------- #

av1 = len(np.where(Data2 == 1)[0])/len(Data2)
av2 = len(np.where(Data2 >= 2)[0])/len(Data2)

tots = []
biases = np.zeros(shape=(2, np.max(clusters)+1))
for i in range(np.max(clusters)+1):
    wh = np.where(clusters_TS == i)[0]
    dat = Data2[wh]
    tot = len(dat[dat >= 1])
    if tot > 0:
        c1 = len(np.where(dat == 1)[0])
        c2 = len(np.where(dat >= 2)[0])
        biases[0, i] = c1/tot
        biases[1, i] = c2/tot
    else:
        biases[0, i] = np.nan
        biases[1, i] = np.nan        
    tots.append(tot)

#%% --------------------------------------------------------------- #
# Save
# ----------------------------------------------------------------- #

path = ('/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/Output/'
        'Louvain6D/Clustercells/')
DF = {}
DF['Cellcode'] = UniLocs
DF['Cluster'] = clusters
DF = pd.DataFrame(DF)
DF.to_csv(path+SUB+'.csv')

path = ('/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/Output/'
        'Louvain6D/Clustermetadata/')
DF2 = {}
DF2['Cluster'] = range(len(biases[0]))
DF2['Explperc'] = biases[1]
DF2['Amountdata'] = tots
pd.DataFrame(DF2).to_csv(path+SUB+'.csv')

DF3 = {}
DF3['Cell'] = Locations
labels = []
for i in tqdm(range(len(Locations))):
    labels.append(int(DF.Cluster[DF.Cellcode == Locations[i]]))
DF3['Cluster'] = labels
pd.DataFrame(DF3).to_pickle(path+'../Locations/'+SUB+'.pkl')

path = ('/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/'
        'Output/Louvain6D/Modularity/')
DF4 = {}
DF4['Modularity'] = [modularity]
pd.DataFrame(DF4).to_csv(path+SUB+'.csv')
