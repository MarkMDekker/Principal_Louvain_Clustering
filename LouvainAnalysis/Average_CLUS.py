# ----------------------------------------------------------------- #
# About script
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import configparser
import pandas as pd
import numpy as np
from tqdm import tqdm
import community as community_louvain
import networkx as nx
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import warnings
warnings.filterwarnings("ignore")
config = configparser.ConfigParser()
config.read('../Pipeline/config.ini')
Path_Data = config['PATHS']['DATA_RAW']
Path_Datasave = config['PATHS']['DATA_SAVE']
Path_Figsave = config['PATHS']['FIG_SAVE']
savepath = '/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/Processed/'

# ----------------------------------------------------------------- #
# Input
# ----------------------------------------------------------------- #

SUBs = ['279418', '279419', '339295', '339296', '346110', '346111']
REG = 'PFC'
REGs = ['HIP', 'PAR', 'PFC']
Name = REGs[0]+'_'+SUBs[0]+'.pkl'
Names = []
for REG in REGs:
    Namei = []
    for SUB in SUBs:
        Name = REG+'_'+SUB+'.pkl'
        Namei.append(Name)
    Names.append(Namei)
Names = np.array(Names)
TAU = 30
RES = 10
space = np.linspace(-12, 12, RES)

# ----------------------------------------------------------------- #
# Read data
# ----------------------------------------------------------------- #

config = configparser.ConfigParser()
config.read('../Pipeline/config.ini')
Path_Data = config['PATHS']['DATA_RAW']
Path_Datasave = config['PATHS']['DATA_SAVE']
Path_Figsave = config['PATHS']['FIG_SAVE']

Series = pd.read_pickle(savepath+'TotalSeries.pkl')
PC1_hip, PC1_par, PC1_pfc, PC2_hip, PC2_par, PC2_pfc, PC1s_hip, PC1s_par, PC1s_pfc, PC2s_hip, PC2s_par, PC2s_pfc, Expl, ROD = np.array(Series).T
mx = np.nanmax([np.abs(PC1_hip), np.abs(PC2_hip), np.abs(PC1_pfc),
                np.abs(PC2_pfc), np.abs(PC1_par), np.abs(PC2_par)])

#%% --------------------------------------------------------------- #
# Locations
# ----------------------------------------------------------------- #


def find_cell(idx, space):
    p1h = np.where(PC1_hip[idx] > space)[0][-1]
    p2h = np.where(PC2_hip[idx] > space)[0][-1]
    p1f = np.where(PC1_pfc[idx] > space)[0][-1]
    p2f = np.where(PC2_pfc[idx] > space)[0][-1]
    p1p = np.where(PC1_par[idx] > space)[0][-1]
    p2p = np.where(PC2_par[idx] > space)[0][-1]
    return str(p1h)+'-'+str(p2h)+'-'+str(p1f)+'-'+str(p2f)+'-'+str(p1p)+'-'+str(p2p)


Locations = []
for i in tqdm(range(len(PC1_hip))):
    try:
        Locations.append(find_cell(i, space))
    except:
        Locations.append(np.nan)
Locations = np.array(Locations)
UniLocs = np.unique(Locations)
UniLocs_sep = []
for i in UniLocs:
    try:
        UniLocs_sep.append(np.array(i.split('-')).astype(int))
    except:
        continue
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

av1 = len(np.where(Expl == 1)[0])/len(Expl)
av2 = len(np.where(Expl >= 2)[0])/len(Expl)

tots = []
biases = np.zeros(shape=(2, np.max(clusters)+1))
for i in range(np.max(clusters)+1):
    wh = np.where(clusters_TS == i)[0]
    dat = Expl[wh]
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
DF.to_csv(path+'Total.csv')

path = ('/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/Output/'
        'Louvain6D/Clustermetadata/')
DF2 = {}
DF2['Cluster'] = range(len(biases[0]))
#DF2['Explperc'] = biases[1]
DF2['Amountdata'] = tots
pd.DataFrame(DF2).to_csv(path+'Total.csv')

DF3 = {}
DF3['Cell'] = Locations
labels = []
for i in tqdm(range(len(Locations))):
    labels.append(int(DF.Cluster[DF.Cellcode == Locations[i]]))
DF3['Cluster'] = labels
pd.DataFrame(DF3).to_pickle(path+'../Locations/Total.pkl')

path = ('/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/'
        'Output/Louvain6D/Modularity/')
DF4 = {}
DF4['Modularity'] = [modularity]
pd.DataFrame(DF4).to_csv(path+'Total.csv')
