# ----------------------------------------------------------------- #
# About script
# Figure 1: illustration of experiment
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import configparser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.gridspec as gridspec

# ----------------------------------------------------------------- #
# Input
# ----------------------------------------------------------------- #

SUBs = ['279418', '279419', '339295', '339296', '340233', '341319',
        '341321', '346110', '346111', '347640', '347641']
REGs = ['HIP', 'PAR', 'PFC']

# ----------------------------------------------------------------- #
# Read data
# ----------------------------------------------------------------- #

config = configparser.ConfigParser()
config.read('../pipeline/config.ini')
Path_Data = config['PATHS']['DATA_RAW']
Path_Datasave = config['PATHS']['DATA_SAVE']
Path_Figsave = ('/Users/mmdekker/Documents/Werk/Figures/Papers/2020_Brain/')

# ----------------------------------------------------------------- #
# Plot
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Plot
# ----------------------------------------------------------------- #

strings = ['(a) Experimental setup',
           '(b) Processing EEG signals']
fig, axes = plt.subplots(figsize=(15, 12))
gs = gridspec.GridSpec(6, 5)
gs.update(wspace=0.025, hspace=0.4)
ax1 = plt.subplot(gs[0:2, :])

ax4 = plt.subplot(gs[2:4, 0])
ax5 = plt.subplot(gs[2:4, 1])
ax6a = plt.subplot(gs[2, 2])
ax6b = plt.subplot(gs[3, 2])
ax7 = plt.subplot(gs[2:4, 3])
ax8 = plt.subplot(gs[2:4, 4])

ax10 = plt.subplot(gs[4:6, 0])
ax11 = plt.subplot(gs[4:6, 1])
ax12 = plt.subplot(gs[4:6, 2])
ax13 = plt.subplot(gs[4:6, 3])
ax14 = plt.subplot(gs[4:6, 4])

# ----------------------------------------------------------------- #
# Experimental setup
# ----------------------------------------------------------------- #

ax1.axis("off")
ax1.text(0.0, 0.98, '(a) Experimental setup', fontsize=14, va='top',
         transform=ax1.transAxes)
ax1.set_xlim([0, 10])
ax1.set_ylim([0, 1])
r1 = mpatches.Rectangle((0.5, 0.4), 0.75, 0.2, lw=2, edgecolor='k', facecolor='silver', alpha=0.5)
r2 = mpatches.Rectangle((1.75, 0.2), 1.25, 0.6, lw=2, edgecolor='k', facecolor='silver', alpha=0.5)
r3 = mpatches.Rectangle((3.5, 0.4), 0.75, 0.2, lw=2, edgecolor='k', facecolor='silver', alpha=0.5)
r4 = mpatches.Rectangle((5.5, 0.4), 0.75, 0.2, lw=2, edgecolor='k', facecolor='silver', alpha=0.5)
r5 = mpatches.Rectangle((6.75, 0.2), 1.25, 0.6, lw=2, edgecolor='k', facecolor='silver', alpha=0.5)
r6 = mpatches.Rectangle((8.5, 0.4), 0.75, 0.2, lw=2, edgecolor='k', facecolor='silver', alpha=0.5)
ax1.add_patch(r1)
ax1.add_patch(r2)
ax1.add_patch(r3)
ax1.add_patch(r4)
ax1.add_patch(r5)
ax1.add_patch(r6)
av1 = 4.75/2
av2 = 14.75/2
k = 0.25
ax1.plot([av1-k, av1+k, av2-k], [0.5, 0.5, 0.5], 'o', ms=15, color='forestgreen')
ax1.plot([av2+k], [0.5], 'o', ms=15, color='tomato')
ax1.text(av2+k, 0.5-0.1, 'New', fontsize=10, ha='center')
ax1.text((1.25-0.5)/2+0.5, 0.42, '5 min', fontsize=12, ha='center', va='bottom')
ax1.text((3-1.75)/2+1.75, 0.22, '10 min', fontsize=12, ha='center', va='bottom')
ax1.text((4.25-3.5)/2+3.5, 0.42, '5 min', fontsize=12, ha='center', va='bottom')
ax1.text(0.75/2+5.5, 0.42, '5 min', fontsize=12, ha='center', va='bottom')
ax1.text(1.25/2+6.75, 0.22, '10 min', fontsize=12, ha='center', va='bottom')
ax1.text(0.75/2+8.5, 0.42, '5 min', fontsize=12, ha='center', va='bottom')

ax1.text((1.25-0.5)/2+0.5, 0.62, 'Home cage 1', fontsize=12, ha='center', va='bottom')
ax1.text((3-1.75)/2+1.75, 0.62, 'Object training', fontsize=12, ha='center', va='bottom')
ax1.text((4.25-3.5)/2+3.5, 0.62, 'Home cage 2', fontsize=12, ha='center', va='bottom')
ax1.text(0.75/2+5.5, 0.62, 'Home cage 3', fontsize=12, ha='center', va='bottom')
ax1.text(1.25/2+6.75, 0.62, 'Object Test', fontsize=12, ha='center', va='bottom')
ax1.text(0.75/2+8.5, 0.62, 'Home cage 4', fontsize=12, ha='center', va='bottom')

k = 0.5
ax1.annotate("", xy=(4.875+k, 0.5), xytext=(4.875-k, 0.5), arrowprops=dict(arrowstyle="->"))
ax1.text(4.875, 0.4, 'One hour'+'\n'+'break', fontsize=12, ha='center', va='top')

# ----------------------------------------------------------------- #
# Processing data
# ----------------------------------------------------------------- #

ax4.text(0.0, 1.1, '(b) Data processing (per brain region)', fontsize=14,
         va='top', transform=ax4.transAxes)
ax5.axis("off")
ax7.axis("off")

# EEG + time window
y = np.random.random(75)
x = np.arange(len(y))
n1 = 10
n2 = 35
av = (n1+n2)/2
ax4.plot(y, c='steelblue', lw=0.3)
ax4.plot(x[(x >= n1) & (x <= n2)], y[(x >= n1) & (x <= n2)], c='steelblue', lw=1.5)
ax4.set_xticks([])
ax4.set_yticks([])
ax4.set_ylabel('EEG signals')
ax4.set_xlabel('Time')
x = np.arange(n1, n2+1)
y1 = np.array([1e3]*len(x))
y0 = np.array([-1e3]*len(x))
ax4.set_ylim(ax4.get_ylim()[0], ax4.get_ylim()[1])
ax4.fill_between(x, y1, y0, where=y1>=y0, alpha=0.15, color='steelblue')
ax4.plot([x[0], x[0]], [y0[0], y1[0]], 'k', lw=0.5)
ax4.plot([x[-1], x[-1]], [y0[0], y1[0]], 'k', lw=0.5)
ax4.text(av, ax4.get_ylim()[1]*0.96, 'Window', ha='center', va='center')
ax4.plot([av]*50, np.arange(50)-48.5, 'k', lw=2)
ax4.plot([av], [0.5], 'o', lw=1, color='k')
ax4.text(av/100+0.1, -0.05, r'$t_0$', ha='center', va='center', transform=ax4.transAxes, fontsize=13, zorder=1e9)

# Arrow in between
ax5.text(0.5, 0.6, 'Apply FFT'+'\n'+'to window data', ha='center', va='bottom')
ax5.text(0.5, 0.4, 'Fit and'+'\n'+r'subtract $a\cdot f^{b}$', ha='center', va='top')
ax5.text(0.5, 0.2, 'Average across'+'\n'+'electrodes in same'+'\n'+'brain region', ha='center', va='top')
ax5.annotate("", xy=(0.75, 0.5), xytext=(0.25, 0.5), arrowprops=dict(arrowstyle="->"))

# FFT + fit + residual
#ax6a.text(056)
ax6a.plot(range(100), 1/(0.01*np.arange(100)))
ax6a.plot(range(100), 1/(0.01*np.arange(100))+np.random.normal(0, 5, size=100), '.')
ax6a.set_xticks([])
ax6a.set_yticks([])
ax6b.set_xticks([])
ax6b.set_yticks([])
ax6a.set_ylabel('Power')
ax6b.set_ylabel('Residual power')
ax6b.set_xlabel('Frequency')
ax6b.plot([0, 100], [0, 0], lw=0.5)
y = np.random.normal(0, 1, size=100)
ax6b.plot(y, '.')
for i in range(len(y)):
    ax6b.plot([i, i], [0, y[i]], c='steelblue', zorder=-1, lw=0.5)
for ax in [ax6a, ax6b]:
    ax.set_xlim(ax6a.get_xlim()[0], ax6a.get_xlim()[1])
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1])
    x = np.arange(78, 85)
    y1 = np.array([1e3]*len(x))
    y0 = np.array([-1e3]*len(x))
    ax.fill_between(x, y1, y0, where=y1>=y0, alpha=0.5, color='silver')
    ax.plot([x[0], x[0]], [y0[0], y1[0]], 'k', lw=0.5)
    ax.plot([x[-1], x[-1]], [y0[0], y1[0]], 'k', lw=0.5)

# Arrow in between
ax7.text(0.5, 0.6, 'Aggregate'+'\n'+'on 2-Hz bands', ha='center', va='bottom')
ax7.text(0.5, 0.4, 'Standardize'+'\n'+'residual series', ha='center', va='top')
ax7.annotate("", xy=(0.75, 0.5), xytext=(0.25, 0.5), arrowprops=dict(arrowstyle="->"))

# Resulting standardized residuals
ax8.pcolormesh(range(100), range(100), np.random.random(size=(100, 100)),
               cmap=plt.cm.RdBu_r, alpha=0.5)
x = np.arange(100)
y1 = np.array([85]*len(x))
y0 = np.array([78]*len(x))
ax8.fill_between(x, y1, y0, where=y1>=y0, alpha=0.8, color='silver')
ax8.plot(x, y1, 'k', lw=0.5)
ax8.plot(x, y0, 'k', lw=0.5)
ax8.set_xticks([])
ax8.set_yticks([])
ax8.set_ylabel('Frequency 2-Hz bands')
ax8.set_xlabel('Time')
ax8.plot([av]*100, range(100), 'k', lw=2)
ax8.plot([av], [81], 'o', lw=1, color='k')
ax8.text(av/100, -0.05, r'$t_0$', ha='center', va='center', transform=ax8.transAxes, fontsize=13, zorder=1e9)

# ----------------------------------------------------------------- #
# Analysing data
# ----------------------------------------------------------------- #

avcov = np.random.random(size=(50, 50))/1.2
for i in range(50):
    avcov[i, i] = 1
ax10.pcolormesh(avcov, alpha=0.25, cmap=plt.cm.Blues)
ax10.set_xticks([])
ax10.set_yticks([])
ax10.set_ylabel('Frequency bands')
ax10.set_xlabel('Frequency bands')
ax10.text(0.02, 0.98, 'Average covariance matrix', transform=ax10.transAxes,
          va='top')
ax10.text(0, 51, '(c) Data analysis', fontsize=14)

ax11.text(0.5, 0.6, 'Apply PCA to'+'\n'+'covariance matrix', ha='center', va='bottom')
ax11.text(0.5, 0.4, 'Construct phase-space'+'\n'+'with PC1 and PC2', ha='center', va='top')
ax11.text(0.5, 0.25, 'Per region', ha='center', va='top')
ax11.text(0.5, 0.15, '(= 6D)', ha='center', va='top')
ax11.annotate("", xy=(0.75, 0.5), xytext=(0.25, 0.5), arrowprops=dict(arrowstyle="->"))
ax11.axis('off')

randwalk1 = (np.arange(15)-7)**2+np.random.random(size=15)*9+np.sin(np.arange(15)/6*2*np.pi)*10
randwalk2 = randwalk1+(np.arange(15)-3)**3+np.random.random(size=15)*9+np.sin(np.arange(15)/6*2*np.pi)*10
randwalk1 = randwalk1/max(randwalk1)-0.12
randwalk2 = randwalk2/max(randwalk2)-0.12
ax12.plot(randwalk1, randwalk2, 'k--')
ax12.set_xticks([])
ax12.set_yticks([])
ax12.set_xlim([-1, 2])
ax12.set_ylim([-1, 2])
for i in np.arange(-3, 5, 0.25):
    ax12.plot([-1e3, 1e3], [i, i], 'silver', lw=0.5, zorder=-1e9)
    ax12.plot([i, i], [-1e3, 1e3], 'silver', lw=0.5, zorder=-1e9)
x = np.arange(0.75, 1.01, 0.01)
y2 = np.array([-0.0]*len(x))
y1 = np.array([-0.25]*len(x))
ax12.fill_between(x, y1, y2, where=y2>=y1, alpha=0.5, color='forestgreen')
y2 = np.array([1]*len(x))
y1 = np.array([0.75]*len(x))
ax12.fill_between(x, y1, y2, where=y2>=y1, alpha=0.5, color='forestgreen')
ax12.annotate("", xy=(0.875, 0.75), xytext=(0.875, 0.), arrowprops=dict(arrowstyle="->"))
ax12.text(0.89, 0.375, r'$\tau$', va='center')
ax12.set_ylabel('PC1 HIP')
ax12.set_xlabel('PC1 PAR')

ax13.text(0.5, 0.6, 'Use conditional'+'\n'+'probabilities', ha='center', va='bottom')
ax13.text(0.5, 0.4, 'of going from'+'\n'+'cell to cell', ha='center', va='top')
ax13.text(0.5, 0.25, 'and apply'+'\n'+'Louvain clustering', ha='center', va='top')
ax13.annotate("", xy=(0.75, 0.5), xytext=(0.25, 0.5), arrowprops=dict(arrowstyle="->"))
ax13.axis('off')


import sys
def windowmean(Data, size):
    ''' Function to compute a running window along a time series '''

    if size == 1:
        return Data
    elif size == 0:
        print('Size 0 not possible!')
        sys.exit()
    else:
        Result = np.zeros(len(Data)) + np.nan
        for i in range(np.int(size/2.), np.int(len(Data) - size/2.)):
            Result[i] = np.nanmean(Data[i-np.int(size/2.):i+np.int(size/2.)])
        return np.array(Result)


ax14.set_xticks([])
ax14.set_yticks([])
ax14.set_xlim([-1, 2])
ax14.set_ylim([-1, 2])
for i in np.arange(-3, 5, 0.25):
    ax14.plot([-1e3, 1e3], [i, i], 'silver', lw=0.5, zorder=-1e9)
    ax14.plot([i, i], [-1e3, 1e3], 'silver', lw=0.5, zorder=-1e9)
for x0 in np.arange(-0.75, 1, 0.25):
    for y0 in np.arange(0.75, 1.75, 0.25):
        x = np.arange(x0, x0+0.26, 0.01)
        y2 = np.array([y0+0.25]*len(x))
        y1 = np.array([y0]*len(x))
        ax14.fill_between(x, y1, y2, where=y2>=y1, alpha=0.5, color='tomato')
for x0 in np.arange(-0.25, 1.5, 0.25):
    for y0 in np.arange(-0.75, 0.25, 0.25):
        x = np.arange(x0, x0+0.26, 0.01)
        y2 = np.array([y0+0.25]*len(x))
        y1 = np.array([y0]*len(x))
        ax14.fill_between(x, y1, y2, where=y2>=y1, alpha=0.5, color='steelblue')
ax14.text(-0.4, 1.2+0.6, 'Cluster A')
ax14.text(0.25, -0.3-0.6, 'Cluster B')
ax14.set_ylabel('PC1 HIP')
ax14.set_xlabel('PC1 PAR')
x1 = windowmean(np.random.random(size=25)*1.8-0.75, 2)
y1 = windowmean(np.random.random(size=25)+0.75, 2)
ax14.plot(x1, y1, '.', c='k')
ax14.plot(x1, y1, c='k')
x2 = windowmean(np.random.random(size=25)*1.8-0.25, 2)
y2 = windowmean(np.random.random(size=25)-0.75, 2)
ax14.plot(x2, y2, '.', c='k')
ax14.plot(x2, y2, c='k')
ax14.plot([x1[-3], x2[3]], [y1[-3], y2[3]], c='k')

fig.tight_layout()
plt.savefig(Path_Figsave+'Figure_1.png', bbox_inches='tight', dpi=200)
