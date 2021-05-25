# ----------------------------------------------------------------- #
# About script
# Figure 3: Example of cluster in 3D space, with trajectory
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
Path_Figsave = ('/Users/mmdekker/Documents/Werk/Figures/Papers/2020_Brain/')
savepath = '/Users/mmdekker/Documents/Werk/Data/Neuro/Processed/'

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
Series = pd.read_pickle(savepath+'TotalSeries.pkl')
SE = np.array(Series).T

# Work with smoothed data:
PC1_hip, PC1_par, PC1_pfc, PC2_hip, PC2_par, PC2_pfc = SE[6:12]
DF_loc = np.array(Series.Cluster_s)
Expl = np.array(Series.Expl)
Cell = np.array(Series.Cell_s)
DF_meta = pd.read_csv(savepath+'ClusterAnalysis_s.csv', index_col=0)
AvExpl = len(Expl[Expl > 1]) / len(Expl)

#%%
# ----------------------------------------------------------------- #
# Plot
# ----------------------------------------------------------------- #

DIM = 0
cmap1 = plt.cm.Blues
cmap2 = plt.cm.Reds

labels = np.copy(DF_loc)

R = 12
N = len(PC1_hip)
C = 4
fig = plt.figure(figsize=(10, 8))
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
Hz2[Hz < 7] = np.nan
Hx2[Hx < 7] = np.nan
Hy2[Hy < 7] = np.nan

B = 50

xedges2 = (xedges[:-1] + xedges[1:])/2.
yedges2 = (yedges[:-1] + yedges[1:])/2.
xe, ye = np.meshgrid(xedges2, yedges2)
ax.contourf(xe, Hy.T, ye, [0, B], zdir='y', offset=R, colors='silver', zorder=5, linewidths=0.5)
ax.contourf(xe, ye, Hz.T, [0, B], zdir='z', offset=-R, colors='silver', zorder=-1, linewidths=0.5)
ax.contourf(Hx.T, xe, ye, [0, B], zdir='x', offset=-R, colors='silver', zorder=-1, linewidths=0.5)
ax.contourf(xe, Hy2.T, ye, 100, zdir='y', offset=R, cmap=cmap1, zorder=-1, vmin=mn, vmax=mx)
ax.contourf(xe, ye, Hz2.T, 100, zdir='z', offset=-R, cmap=cmap1, zorder=-1, vmin=mn, vmax=mx)
ax.contourf(Hx2.T, xe, ye, 100, zdir='x', offset=-R, cmap=cmap1, zorder=-1, vmin=mn, vmax=mx)
ax.set_xticks(range(-R, R+1))
ax.set_yticks(range(-R, R+1))
ax.set_zticks(range(-R, R+1))

# One cluster
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

# Two clusters
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

ax.set_xlabel('HIP, PC1', rotation=-35)
ax.set_xlim(-R, R)
ax.set_ylabel('PFC, PC1', rotation=35)
ax.set_ylim(-R, R)
ax.set_zlabel('PAR, PC1', rotation=-90)
ax.set_zlim(-R, R)
Ni = int(DF_meta.AmountData[DF_meta.Clusters == C])
#ax.set_title('PC'+str(DIM+1)+', cluster '+str(C)+', '+str(len(DF_cell[DF_cell.Cluster == C]))+' cells, '+str(Ni)+' datapoints ('+str(np.round(Ni/N*100, 1))+'%) in cluster'+'\n'+'Bias: '+str(np.round(np.float(DF_meta.Explbias[DF_meta.Cluster == C]), 2)), fontsize=8)
ax.tick_params(axis='both', which='major', labelsize=7, pad=-3)
ax.tick_params(axis='both', which='minor', labelsize=7, pad=-3)
ax.set_xticks(space)
ax.set_yticks(space)
ax.set_zticks(space)
ax.set_xticklabels([-12, '', '', -4, '', '', 4, '', '', 12], rotation=-25)
ax.set_yticklabels([-12, '', '', -4, '', '', 4, '', '', 12], rotation=25)
ax.set_zticklabels([-12, '', '', -4, '', '', 4, '', '', 12])
ax.dist = 10.4
#fig.tight_layout()

# Colormap for clusterdata
axc = fig.add_axes([0.95, 0.15, 0.02, 0.7])
norm = mpl.colors.Normalize(vmin=mnc, vmax=mxc)
cb2 = mpl.colorbar.ColorbarBase(axc, cmap=cmap2, norm=norm,
                                orientation='vertical')
cb2.set_label('Density clusterdata', fontsize=10)
axc.yaxis.set_label_position('right')
axc.yaxis.tick_right()
axc.tick_params(labelsize=8)

# Colormap for all data
axc = fig.add_axes([1.07, 0.15, 0.02, 0.7])
norm = mpl.colors.Normalize(vmin=mn, vmax=mx)
cb2 = mpl.colorbar.ColorbarBase(axc, cmap=cmap1, norm=norm,
                                orientation='vertical')
cb2.set_label('Density all data', fontsize=10)
axc.yaxis.set_label_position('right')
axc.yaxis.tick_right()
axc.tick_params(labelsize=8)

plt.savefig(Path_Figsave+'Figure_3.png', bbox_inches='tight', dpi=200)
