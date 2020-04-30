# --------------------------------------------------------------------- #
# Preambule
# --------------------------------------------------------------------- #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import matplotlib as mpl
from scipy.ndimage.filters import gaussian_filter
datadir = '/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/ObjectTest.mat'
path_pdata = ('/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/'
              'ProcessedData/')
path_figsave = ('/Users/mmdekker/Documents/Werk/Figures/Figures_Side/Brain/'
                'Pics_newest/')

# --------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------- #

Str = 'HIP'
path_figsave = path_figsave + Str
res = 200

# --------------------------------------------------------------------- #
# Read raw data
# --------------------------------------------------------------------- #

mat = scipy.io.loadmat(datadir)
DataAll = mat['EEG'][0][0]
Time = DataAll[14]
Data_dat = DataAll[15]
Data_re1 = DataAll[17]
Data_re2 = DataAll[18]
Data_re3 = DataAll[19]
Data_re4 = DataAll[25]
Data_re5 = DataAll[30]
Data_re6 = DataAll[39]
Data_vid = DataAll[40].T
ExploringNew = np.where(Data_vid[2] == 1)[0]
ExploringOld = np.where(Data_vid[3] == 1)[0]
NotExploring = np.arange(len(Data_vid[0]))[(Data_vid[2] != 1) & (Data_vid[3] != 1)]
TS_new = ExploringNew[ExploringNew < 599000]
TS_old = ExploringOld[ExploringOld < 599000]
TS_not = NotExploring[NotExploring < 599000]

# --------------------------------------------------------------------- #
# Get PCs
# --------------------------------------------------------------------- #

PCs = pd.read_pickle(path_pdata+'PCs/'+Str+'/600000_0.pkl')
PCx = np.array(PCs)[0]
PCy = np.array(PCs)[1]

# --------------------------------------------------------------------- #
# Extract parts of PCs
# --------------------------------------------------------------------- #

X_new = PCx[ExploringNew[ExploringNew < 599000]]
Y_new = PCy[ExploringNew[ExploringNew < 599000]]

X_old = PCx[ExploringOld[ExploringOld < 599000]]
Y_old = PCy[ExploringOld[ExploringOld < 599000]]

X_not = PCx[NotExploring[NotExploring < 599000]]
Y_not = PCy[NotExploring[NotExploring < 599000]]

X_tot = PCx
Y_tot = PCy

# --------------------------------------------------------------------- #
# Calculate histograms in terms of the grid cells
# --------------------------------------------------------------------- #

space = np.linspace(np.floor(np.min([PCx, PCy])),
                    np.floor(np.max([PCx, PCy]))+1, res)
H_new, xedges, yedges = np.histogram2d(X_new, Y_new, bins=[space, space])
H_old, xedges, yedges = np.histogram2d(X_old, Y_old, bins=[space, space])
H_not, xedges, yedges = np.histogram2d(X_not, Y_not, bins=[space, space])
H_tot, xedges, yedges = np.histogram2d(X_tot, Y_tot, bins=[space, space])
xedges2 = (xedges[:-1] + xedges[1:])/2.
yedges2 = (yedges[:-1] + yedges[1:])/2.

# --------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------- #

Hs = [H_not, H_old, H_new]
strings = ['(a) Not exploring', '(b) Exploring old', '(c) Exploring new']
fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
for i in range(3):
    ax = axes[i]
    ax.text(0.02, 0.98, strings[i], transform=ax.transAxes, va='top',
            fontsize=12)
    H = Hs[i].T / (H_tot.T+1e-9)
    H2 = gaussian_filter(H, 0.5)
    ax.contourf(xedges2, yedges2, Hs[i].T / H_tot.T, zorder=-1,
                cmap=plt.cm.Blues)
    ax.contour(xedges2, yedges2, H2, [0.25], zorder=-1,
               colors='k', linewidths=1)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    if i == 0:
        ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_ylim([-7, 7])
    ax.set_xlim([-7, 7])
fig.tight_layout()

axc = fig.add_axes([0.35, -0.035, 0.35, 0.03])
norm = mpl.colors.Normalize(vmin=0, vmax=1)
cb1 = mpl.colorbar.ColorbarBase(axc, cmap=plt.cm.Blues, norm=norm,
                                orientation='horizontal')
cb1.set_label('Rodent activity in percentage of total realizations in grid cell', fontsize=15)
cb1.set_ticks(np.arange(0.0, 1.1, 0.25))
axc.tick_params(labelsize=13)

plt.savefig(path_figsave+'/Interpretation.png', bbox_inches='tight', dpi=200)
