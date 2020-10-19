# ----------------------------------------------------------------- #
# About script
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
figpath = ('/Users/mmdekker/Documents/Werk/Figures/Figures_Side/Brain/'
           'Pics_20201009/')

# ----------------------------------------------------------------- #
# Input
# ----------------------------------------------------------------- #

SUB = '279418'
# 346110
# 339295
# 279418
RES = 10

# ----------------------------------------------------------------- #
# Read data
# ----------------------------------------------------------------- #

space = np.linspace(-12, 12, RES)
path = ('/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/Output/'
        'Louvain6D/Clustermetadata/')
DF_meta = pd.read_csv(path+SUB+'.csv')
DF_cell = pd.read_csv(path+'/../Clustercells/'+SUB+'.csv')

AvExpl = np.sum(DF_meta.Explperc*DF_meta.Amountdata)/np.sum(DF_meta.Amountdata)

PCs_hip = np.array(pd.read_pickle(path+'/../../PCs/F_HIP_'+SUB+'.pkl'))[:2]
PCs_par = np.array(pd.read_pickle(path+'/../../PCs/F_PAR_'+SUB+'.pkl'))[:2]
PCs_pfc = np.array(pd.read_pickle(path+'/../../PCs/F_PFC_'+SUB+'.pkl'))[:2]

DF_loc = pd.read_pickle(path+'../Locations/'+SUB+'.pkl')

#%% --------------------------------------------------------------- #
# Plot biases
# ----------------------------------------------------------------- #


def renormalize(vec):
    vec = vec - 0.5
    vec[vec < 0] = 0
    vec[vec > 1] = 1
    return vec


cols = ['steelblue', 'tomato', 'forestgreen']
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
ax.text(0.02, 0.95, 'Average % exploring '+str(np.round(100*AvExpl, 2))+'%',
        transform=ax.transAxes, va='top')
ax.plot(DF_meta.Explperc/AvExpl, c='k', label='Exploring', zorder=-1)
ax.scatter(range(len(DF_meta)), DF_meta.Explperc/AvExpl,
           c=renormalize(DF_meta.Explperc/AvExpl),
           cmap=plt.cm.coolwarm, edgecolor='k', s=100)
ax.plot([-1e3, 1e3], [1, 1], 'k:', lw=1, zorder=-1)
ax.set_xlim([-1, len(DF_meta)+1])
ax2.set_xlabel('Index cluster')
ax.set_ylabel('Bias towards exploring')
ax2.scatter(range(len(DF_meta)), DF_meta.Amountdata,
            c=renormalize(DF_meta.Explperc/AvExpl),
            cmap=plt.cm.coolwarm, edgecolor='k', s=100)
ax2.plot(DF_meta.Amountdata, 'k', zorder=-1)
ax2.set_ylabel('Amount of data in cluster')
fig.tight_layout()
plt.savefig(figpath+'Bias_'+SUB+'.png', bbox_inches='tight', dpi=200)

#%% --------------------------------------------------------------- #
# Relate size and absolute bias
# ----------------------------------------------------------------- #

fig, ax = plt.subplots(figsize=(6, 6))
ax.semilogy(np.abs(1-DF_meta.Explperc/AvExpl), DF_meta.Amountdata, '.')
ax.text(0.95, 0.95, 'r = '+str(np.round(np.corrcoef(np.abs(1-DF_meta.Explperc/AvExpl), np.log(DF_meta.Amountdata))[0, 1], 2)), transform=ax.transAxes, ha='right', va='top')
ax.set_ylabel('Size cluster (# datapoints)')
ax.set_xlabel('Absolute bias')
fig.tight_layout()
plt.savefig(figpath+'Relation_'+SUB+'.png', bbox_inches='tight', dpi=200)

#%% --------------------------------------------------------------- #
# Make 3D plot of situation of biases
# ----------------------------------------------------------------- #

DIM = 0

PC1_hip = PCs_hip[DIM]
PC1_pfc = PCs_pfc[DIM]
PC1_par = PCs_par[DIM]
labels = np.array(DF_loc.Cluster)

R = 12
N = len(PC1_hip)
C = 30
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.view_init(45, -45)
X, Y, Z = PC1_hip[:N], PC1_pfc[:N], PC1_par[:N]
#ax.scatter(X, Y, Z, s=0.2, c='silver', alpha=0.01, zorder=0)
#ax.scatter(X[labels[:N]==C], Y[labels[:N]==C], Z[labels[:N]==C], s=0.2,
#           c='forestgreen', zorder=1e9)
space2 = np.linspace(-12, 12, 200)

Hz, xedges, yedges = np.histogram2d(X, Y, bins=[space2, space2])
Hx, xedges, yedges = np.histogram2d(Y, Z, bins=[space2, space2])
Hy, xedges, yedges = np.histogram2d(X, Z, bins=[space2, space2])
Hz[Hz == 0] = np.nan
Hx[Hx == 0] = np.nan
Hy[Hy == 0] = np.nan

xedges2 = (xedges[:-1] + xedges[1:])/2.
yedges2 = (yedges[:-1] + yedges[1:])/2.
xe, ye = np.meshgrid(xedges2, yedges2)
# ax.contourf(xe, Hy.T, ye, 6, zdir='y', offset=R, colors='k', zorder=-1)
# ax.contourf(xe, ye, Hz.T, 6, zdir='z', offset=-R, colors='k', zorder=-1)
# ax.contourf(Hx.T, xe, ye, 6, zdir='x', offset=-R, colors='k', zorder=-1)
ax.contourf(xe, Hy.T, ye, 100, zdir='y', offset=R, cmap=plt.cm.Blues, zorder=-1)
ax.contourf(xe, ye, Hz.T, 100, zdir='z', offset=-R, cmap=plt.cm.Blues, zorder=-1)
ax.contourf(Hx.T, xe, ye, 100, zdir='x', offset=-R, cmap=plt.cm.Blues, zorder=-1)
ax.set_xticks(range(-R, R+1))
ax.set_yticks(range(-R, R+1))
ax.set_zticks(range(-R, R+1))


wh = np.where(labels[:N] == C)[0]
Xi, Yi, Zi = PC1_hip[:N][wh], PC1_pfc[:N][wh], PC1_par[:N][wh]

Hz, xedges, yedges = np.histogram2d(Xi, Yi, bins=[space2, space2])
Hx, xedges, yedges = np.histogram2d(Yi, Zi, bins=[space2, space2])
Hy, xedges, yedges = np.histogram2d(Xi, Zi, bins=[space2, space2])
Hz[Hz == 0] = np.nan
Hx[Hx == 0] = np.nan
Hy[Hy == 0] = np.nan
xedges2 = (xedges[:-1] + xedges[1:])/2.
yedges2 = (yedges[:-1] + yedges[1:])/2.
xe, ye = np.meshgrid(xedges2, yedges2)
col = plt.cm.rainbow(C/np.max(labels))
#ax.contourf(xe, Hy.T, ye, 6, zdir='y', offset=R, colors='forestgreen', zorder=-1)
#ax.contourf(xe, ye, Hz.T, 6, zdir='z', offset=-R, colors='forestgreen', zorder=-1)
#ax.contourf(Hx.T, xe, ye, 6, zdir='x', offset=-R, colors='forestgreen', zorder=-1)
ax.contourf(xe, Hy.T, ye, 100, zdir='y', offset=R, cmap=plt.cm.gist_heat_r, zorder=-1)
ax.contourf(xe, ye, Hz.T, 100, zdir='z', offset=-R, cmap=plt.cm.gist_heat_r, zorder=-1)
ax.contourf(Hx.T, xe, ye, 100, zdir='x', offset=-R, cmap=plt.cm.gist_heat_r, zorder=-1)

ax.set_xlabel('HIP')
ax.set_xlim(-R, R)
ax.set_ylabel('PFC')
ax.set_ylim(-R, R)
ax.set_zlabel('PAR')
ax.set_zlim(-R, R)
ax.set_title('PC'+str(DIM+1)+', rodent '+SUB+', cluster '+str(C)+', '+str(len(DF_cell[DF_cell.Cluster == C]))+' cells, '+str(int(DF_meta.Amountdata[DF_meta.Cluster == C]))+' datapoints', fontsize=8)
ax.tick_params(axis='both', which='major', labelsize=7)
ax.tick_params(axis='both', which='minor', labelsize=7)
fig.tight_layout()
plt.savefig(figpath+'ExClus_'+SUB+'.png', bbox_inches='tight', dpi=200)
