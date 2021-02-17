# ----------------------------------------------------------------- #
# About script
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy.stats
import matplotlib.pyplot as plt
path = '/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/Processed/'

# ----------------------------------------------------------------- #
# Read data
# ----------------------------------------------------------------- #

space = np.linspace(-12, 12, 10)

# Cell = pd.read_csv(path+'Cells.csv')
Cell_s = pd.read_csv(path+'Cells_s.csv', index_col=0)
# Cell_sh = pd.read_csv(path+'Cells_sh.csv')

# DF = pd.read_csv(path+'ClusterAnalysis.csv')
DF_s = pd.read_csv(path+'ClusterAnalysis_s.csv', index_col=0)
# DF_sh = pd.read_csv(path+'ClusterAnalysis_sh.csv')

# N = len(DF)
N_s = len(DF_s)
# N_sh = len(DF_sh)

Series = pd.read_pickle(path+'TotalSeries.pkl')
# Clus = np.array(Series.Cluster)
Clus_s = np.array(Series.Cluster_s)
# Clus_sh = np.array(Series.Cluster_sh)

#%%
# ----------------------------------------------------------------- #
# Centroids (based on average 6D in points, not cells)
# ----------------------------------------------------------------- #

centroids = np.zeros(shape=(N_s, 6))
for i in tqdm(range(N_s)):
    wh = np.where(Clus_s == i)[0]
    dat = np.array(Series[['PC1s_hip', 'PC1s_par', 'PC1s_pfc', 'PC2s_hip', 'PC2s_par', 'PC2s_pfc']][Series.index.isin(wh)])
    centroid = np.nanmean(dat, axis=0)
    centroids[i] = centroid

#%%
Mcentrdiff = np.zeros(shape=(N_s, N_s))+np.nan
for i in range(N_s):
    for j in range(N_s):
        Mcentrdiff[i, j] = np.sqrt(np.nansum((centroids[i]-centroids[j])**2))

# ----------------------------------------------------------------- #
# Exploration bias differences
# ----------------------------------------------------------------- #

biases = np.array(DF_s.Explbias)
Mbiasdiff = np.zeros(shape=(N_s, N_s))
for i in range(N_s):
    for j in range(N_s):
        Mbiasdiff[i, j] = np.abs(biases[i] - biases[j])

#%%
# ----------------------------------------------------------------- #
# Plot correlations
# ----------------------------------------------------------------- #

tuples = np.zeros(shape=(2, int((N_s**2-N_s)/2)))

a = 0
for i in range(N_s):
    for j in range(N_s):
        if i > j:
            tuples[0, a] = Mcentrdiff[i, j]
            tuples[1, a] = Mbiasdiff[i, j]
            a += 1

#%%
fig, ax = plt.subplots(figsize=(9, 9))
ax.plot(tuples[0], tuples[1],'.')
ax.text(0.03, 0.97, 'Spearman r: '+str(scipy.stats.spearmanr(tuples[0], tuples[1])[0].round(2)), va='top', transform=ax.transAxes)
ax.text(0.03, 0.93, 'Pearson r: '+str(np.corrcoef(tuples[0], tuples[1])[0][1].round(2)), va='top', transform=ax.transAxes)
ax.set_xlabel('Centroid differences')
ax.set_ylabel('Bias differences')