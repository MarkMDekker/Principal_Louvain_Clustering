# ----------------------------------------------------------------- #
# About script
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import configparser
import pandas as pd
import scipy.stats
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
figpath = ('/Users/mmdekker/Documents/Werk/Figures/Figures_Side/Brain/'
           'Pics_20201020/')

# ----------------------------------------------------------------- #
# Input
# ----------------------------------------------------------------- #

RES = 10
TAU = 30
space = np.linspace(-12, 12, RES)

# ----------------------------------------------------------------- #
# Read data
# ----------------------------------------------------------------- #

config = configparser.ConfigParser()
config.read('../Pipeline/config.ini')
Path_Data = config['PATHS']['DATA_RAW']
Path_Datasave = config['PATHS']['DATA_SAVE']
Path_Figsave = ('/Users/mmdekker/Documents/Werk/Figures/Papers/2020_Brain/')
Series = pd.read_pickle(Path_Datasave+'PCs/Total.pkl')
PC1_hip, PC2_hip, PC1_par, PC2_par, PC1_pfc, PC2_pfc, Expl = np.array(Series).T
#%%
path = ('/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/Output/'
        'Louvain6D/Clustermetadata/')
DF_meta = pd.read_csv(path+'Total.csv')
DF_cell = pd.read_csv(path+'/../Clustercells/Total.csv')
AvExpl = len(Expl[Expl > 1]) / len(Expl)

DF_loc = pd.read_pickle(path+'../Locations/Total.pkl')

# --------------------------------------------------------------- #
# Relate size and absolute bias
# ----------------------------------------------------------------- #

X = np.abs(1-DF_meta.Explperc/AvExpl)
Y = DF_meta.Amountdata

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.semilogy(X, Y, '.')
pearsonr = np.corrcoef(X, Y)[0, 1]
pearsonrl = np.corrcoef(X, np.log(Y))[0, 1]
ax1.text(0.95, 0.95, 'r = '+str(np.round(pearsonr, 2)),
         transform=ax1.transAxes, ha='right', va='top')
ax1.text(0.95, 0.9, 'rl = '+str(np.round(pearsonrl, 2)),
         transform=ax1.transAxes, ha='right', va='top')
ax1.set_ylabel('Size cluster (# datapoints)')
ax1.set_xlabel('Absolute bias')

ranky = np.argsort(X).argsort()
rankx = np.argsort(Y).argsort()
ax2.plot(rankx, ranky, '.')
ax2.set_ylabel('Rank (size cluster)')
ax2.set_xlabel('Rank (absolute bias)')

spearmanr = scipy.stats.spearmanr(X, Y)[0]
ax2.text(0.95, 0.95, 'Sr = '+str(np.round(spearmanr, 2)),
         transform=ax2.transAxes, ha='right', va='top')
fig.tight_layout()
plt.savefig(Path_Figsave+'Figure_6.png', bbox_inches='tight', dpi=200)