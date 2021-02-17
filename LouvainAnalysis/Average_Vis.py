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
Path_Figsave = config['PATHS']['FIG_SAVE']
Series = pd.read_pickle(Path_Datasave+'PCs/Total.pkl')
PC1_hip, PC2_hip, PC1_par, PC2_par, PC1_pfc, PC2_pfc, Expl, ROD = np.array(Series).T

path = ('/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/Output/'
        'Louvain6D/Clustermetadata/')
DF_meta = pd.read_csv(path+'Total_analysis.csv')
DF_cell = pd.read_csv(path+'/../Clustercells/Total.csv')
AvExpl = len(Expl[Expl > 1]) / len(Expl)

DF_loc = pd.read_pickle(path+'../Locations/Total.pkl')

#%% --------------------------------------------------------------- #
# Plot biases
# ----------------------------------------------------------------- #


def renormalize(vec):
    vec = vec - 0.5
    vec[vec < 0] = 0
    vec[vec > 1] = 1
    return vec


cols = ['steelblue', 'tomato', 'forestgreen']
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
ax.text(0.02, 0.95, 'Average % exploring '+str(np.round(100*AvExpl, 2))+'%',
        transform=ax.transAxes, va='top')
ax.plot(DF_meta.Explbias/AvExpl, c='k', label='Exploring', zorder=-1)
ax.scatter(range(len(DF_meta)), DF_meta.Explbias/AvExpl,
           c=renormalize(DF_meta.Explbias/AvExpl),
           cmap=plt.cm.coolwarm, edgecolor='k', s=100)
ax.plot([-1e3, 1e3], [1, 1], 'k:', lw=1, zorder=-1)
ax.set_xlim([-1, len(DF_meta)+1])
ax2.set_xlabel('Index cluster')
ax.set_ylabel('Bias towards exploring')
ax2.scatter(range(len(DF_meta)), DF_meta.Amountdata,
            c=renormalize(DF_meta.Explbias/AvExpl),
            cmap=plt.cm.coolwarm, edgecolor='k', s=100)
ax2.semilogy(DF_meta.Amountdata, 'k', zorder=-1)
ax2.set_ylabel('Amount of data in cluster')
ax2.set_ylim([ax2.get_ylim()[0], ax2.get_ylim()[1]])
ax2.set_xlim([ax2.get_xlim()[0], ax2.get_xlim()[1]])
for i in range(10):
    ax2.plot([-1e3, 1e3], [10**i, 10**i], 'k:', lw=0.5, zorder=-1)
fig.tight_layout()
plt.savefig(figpath+'Bias_Av.png', bbox_inches='tight', dpi=200)

#%% --------------------------------------------------------------- #
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
plt.savefig(figpath+'Relation_Av.png', bbox_inches='tight', dpi=200)

#%% --------------------------------------------------------------- #
# Make 3D plot of situation of biases
# ----------------------------------------------------------------- #

DIM = 0
cmap1 = plt.cm.Reds
cmap2 = plt.cm.terrain

labels = np.array(DF_loc.Cluster)

R = 12
N = len(PC1_hip)
C = 36
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.view_init(45, -45)
X, Y, Z = PC1_hip[:N], PC1_pfc[:N], PC1_par[:N]
space2 = np.linspace(-12, 12, 200)

Hz, xedges, yedges = np.histogram2d(X, Y, bins=[space2, space2])
Hx, xedges, yedges = np.histogram2d(Y, Z, bins=[space2, space2])
Hy, xedges, yedges = np.histogram2d(X, Z, bins=[space2, space2])
Hz[Hz == 0] = np.nan
Hx[Hx == 0] = np.nan
Hy[Hy == 0] = np.nan
mx = np.nanmax([Hx, Hy, Hz])
mn = np.nanmin([Hx, Hy, Hz])
Hx2, Hy2, Hz2 = np.array([Hx, Hy, Hz])
Hz2[Hz < 15] = np.nan
Hx2[Hx < 15] = np.nan
Hy2[Hy < 15] = np.nan

xedges2 = (xedges[:-1] + xedges[1:])/2.
yedges2 = (yedges[:-1] + yedges[1:])/2.
xe, ye = np.meshgrid(xedges2, yedges2)
ax.contourf(xe, Hy.T, ye, [0, 100], zdir='y', offset=R, colors='grey', zorder=5, linewidths=0.5)
ax.contourf(xe, ye, Hz.T, [0, 100], zdir='z', offset=-R, colors='grey', zorder=-1, linewidths=0.5)
ax.contourf(Hx.T, xe, ye, [0, 100], zdir='x', offset=-R, colors='grey', zorder=-1, linewidths=0.5)
ax.contourf(xe, Hy2.T, ye, 100, zdir='y', offset=R, cmap=cmap1, zorder=-1, vmin=mn, vmax=mx)
ax.contourf(xe, ye, Hz2.T, 100, zdir='z', offset=-R, cmap=cmap1, zorder=-1, vmin=mn, vmax=mx)
ax.contourf(Hx2.T, xe, ye, 100, zdir='x', offset=-R, cmap=cmap1, zorder=-1, vmin=mn, vmax=mx)
ax.set_xticks(range(-R, R+1))
ax.set_yticks(range(-R, R+1))
ax.set_zticks(range(-R, R+1))


wh = np.where(labels[:N] == C)[0]
Xi, Yi, Zi = PC1_hip[:N][wh], PC1_pfc[:N][wh], PC1_par[:N][wh]

Hz, xedges, yedges = np.histogram2d(Xi, Yi, bins=[space2, space2])
Hx, xedges, yedges = np.histogram2d(Yi, Zi, bins=[space2, space2])
Hy, xedges, yedges = np.histogram2d(Xi, Zi, bins=[space2, space2])
Hz[Hz <= np.nanpercentile(Hz, 99)] = np.nan
Hx[Hx <= np.nanpercentile(Hx, 99)] = np.nan
Hy[Hy <= np.nanpercentile(Hy, 99)] = np.nan
mxc = np.nanmax([Hx, Hy, Hz])
mnc = np.nanmin([Hx, Hy, Hz])
xedges2 = (xedges[:-1] + xedges[1:])/2.
yedges2 = (yedges[:-1] + yedges[1:])/2.
xe, ye = np.meshgrid(xedges2, yedges2)
col = plt.cm.rainbow(C/np.max(labels))
ax.contourf(xe, Hy.T, ye, 100, zdir='y', offset=R, cmap=cmap2, zorder=-1, vmin=mnc, vmax=mxc)
ax.contourf(xe, ye, Hz.T, 100, zdir='z', offset=-R, cmap=cmap2, zorder=-1, vmin=mnc, vmax=mxc)
ax.contourf(Hx.T, xe, ye, 100, zdir='x', offset=-R, cmap=cmap2, zorder=-1, vmin=mnc, vmax=mxc)

ax.set_xlabel('HIP', rotation=-35)
ax.set_xlim(-R, R)
ax.set_ylabel('PFC', rotation=35)
ax.set_ylim(-R, R)
ax.set_zlabel('PAR', rotation=-90)
ax.set_zlim(-R, R)
Ni = int(DF_meta.Amountdata[DF_meta.Cluster == C])
ax.set_title('PC'+str(DIM+1)+', cluster '+str(C)+', '+str(len(DF_cell[DF_cell.Cluster == C]))+' cells, '+str(Ni)+' datapoints ('+str(np.round(Ni/N*100, 1))+'%) in cluster'+'\n'+'Bias: '+str(np.round(np.float(DF_meta.Explperc[DF_meta.Cluster == C])/AvExpl, 2)), fontsize=8)
ax.tick_params(axis='both', which='major', labelsize=7)
ax.tick_params(axis='both', which='minor', labelsize=7)
fig.tight_layout()

# Colormap for clusterdata
axc = fig.add_axes([1.05, 0.15, 0.02, 0.7])
norm = mpl.colors.Normalize(vmin=mnc, vmax=mxc)
cb2 = mpl.colorbar.ColorbarBase(axc, cmap=cmap2, norm=norm,
                                orientation='vertical')
cb2.set_label('Density clusterdata', fontsize=10)
axc.yaxis.set_label_position('right')
axc.yaxis.tick_right()
axc.tick_params(labelsize=8)

# Colormap for all data
axc = fig.add_axes([1.2, 0.15, 0.02, 0.7])
norm = mpl.colors.Normalize(vmin=mn, vmax=mx)
cb2 = mpl.colorbar.ColorbarBase(axc, cmap=cmap1, norm=norm,
                                orientation='vertical')
cb2.set_label('Density all data', fontsize=10)
axc.yaxis.set_label_position('right')
axc.yaxis.tick_right()
axc.tick_params(labelsize=8)

plt.savefig(figpath+'ExClus_Av_'+str(C)+'.png', bbox_inches='tight', dpi=200)
