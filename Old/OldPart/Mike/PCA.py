# --------------------------------------------------------------------- #
# Preambule
# --------------------------------------------------------------------- #

import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
from sklearn.cluster import KMeans
import matplotlib as mpl
import matplotlib.pyplot as plt
exec(open('/Users/mmdekker/Documents/Werk/Scripts/__Other/BrainProject/Mike/Functions.py').read())
from numba import jit

@jit
def windowmean(Data,size):
    if size==1: return Data
    elif size==0:
        print('Size 0 not possible!')
        sys.exit()
    else:
        Result=np.zeros(len(Data))+np.nan
        for i in range(np.int(size/2.),np.int(len(Data)-size/2.)):
            Result[i]=np.nanmean(Data[i-np.int(size/2.):i+np.int(size/2.)])
        return np.array(Result)
    
# --------------------------------------------------------------------- #
# Read
# --------------------------------------------------------------------- #

Cmats = np.load('/Users/mmdekker/Documents/Werk/Scripts/__Other/BrainProject/Mike/ProcData/Cmats.npy')
C = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Scripts/__Other/BrainProject/Mike/ProcData/C.pkl'))

# --------------------------------------------------------------------- #
# Convert
# --------------------------------------------------------------------- #

TimeSeries = []
for i in tqdm(range(len(Cmats[0])),file=sys.stdout):
    for j in range(len(Cmats[0][0])):
        S = Cmats[:,i,j]
        #S = windowmean(S,26)
        S = S - np.nanmean(S)
        S = S / np.nanstd(S)
        #S = S[13:-13]
        TimeSeries.append(S)
TimeSeries = np.array(TimeSeries)
TimeSeries[np.isnan(TimeSeries)] = 0
#%%
# --------------------------------------------------------------------- #
# PCA
# --------------------------------------------------------------------- #

vals,vecs = np.linalg.eig(TimeSeries.dot(TimeSeries.T))
idx = vals.argsort()[::-1]
vals_num = vals/np.sum(vals)
vals = vals[idx]
vecs = vecs[:,idx]
EOFs = []
PCs = []
for i in tqdm(range(50),file=sys.stdout):
    EOFs.append(vecs[:,i])
    PCs.append(vecs[:,i].dot(TimeSeries))

#%%
# Rescale Synchronization
Mvec = np.copy(C)#[13:-13])
Mvec = Mvec - np.min(Mvec)
Mvec = Mvec / np.max(Mvec)
Mvec = Mvec[:,0]

WM = 1
x = 1
y = 2
fig,ax = plt.subplots(figsize=(8,8))
ax.plot(windowmean(PCs[x],WM),windowmean(PCs[y],WM),'k',lw=0.2)
c=ax.scatter(windowmean(PCs[x],WM),windowmean(PCs[y],WM),c=plt.cm.rainbow(Mvec),zorder=5,s=25)
ax.set_xlabel('Principal Component '+str(x+1)+' ('+str(int(vals_num[x]*100))+'%)')
ax.set_ylabel('Principal Component '+str(y+1)+' ('+str(int(vals_num[y]*100))+'%)')
axc     = fig.add_axes([1., 0.1, 0.025, 0.8])
norm    = mpl.colors.Normalize(vmin=np.min(C), vmax=np.max(C))
cb1     = mpl.colorbar.ColorbarBase(axc, cmap=plt.cm.rainbow,norm=norm,extend='both',orientation='vertical')
cb1.set_label('Correlation matrix average',fontsize=15)
axc.tick_params(labelsize=13)
fig.tight_layout()

#%%
Cmat_eigs = np.zeros(shape=(10,55,55))
for eig in range(10):
    for i in range(55):
        Cmat_eigs[eig,i] = EOFs[eig][i*55:i*55+55]

cmap = plt.cm.seismic

fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10,10),sharex=True,sharey=True)
ax1.pcolormesh(Cmat_eigs[0],cmap=cmap,vmin=np.min(Cmat_eigs),vmax=np.max(Cmat_eigs))
ax2.pcolormesh(Cmat_eigs[1],cmap=cmap,vmin=np.min(Cmat_eigs),vmax=np.max(Cmat_eigs))
ax3.pcolormesh(Cmat_eigs[2],cmap=cmap,vmin=np.min(Cmat_eigs),vmax=np.max(Cmat_eigs))
ax4.pcolormesh(Cmat_eigs[3],cmap=cmap,vmin=np.min(Cmat_eigs),vmax=np.max(Cmat_eigs))

axc     = fig.add_axes([0.2, 0.05, 0.6, 0.025])
norm    = mpl.colors.Normalize(vmin=np.min(Cmat_eigs), vmax=np.max(Cmat_eigs))
cb1     = mpl.colorbar.ColorbarBase(axc, cmap=cmap,norm=norm,extend='both',orientation='horizontal')
cb1.set_label('Eigenvector Coefficients',fontsize=15)
axc.tick_params(labelsize=13)
fig.tight_layout

#%%
def ClusterAndPlot(ax,Matrix,n,vmin,vmax,cmap):
    global allclusters
    kmeans = KMeans(n_clusters=n, random_state=0).fit(Matrix)
    Labels = kmeans.labels_
    DF = {}
    DF['Range'] = range(55)
    DF['Labels'] = Labels
    DF = pd.DataFrame(DF)
    DF = DF.sort_values(by=['Labels'])
    Order = np.array(DF.Range)
    allclusters.append(Labels)
    
    Matrix2 = np.zeros(shape=(55,55))
    for i in range(55):
        for j in range(55):
            Matrix2[i,j] = Matrix[Order[i],Order[j]]
        
    ax.pcolormesh(Matrix2,vmin=vmin,vmax=vmax,cmap=cmap)
    ax.set_xlim([0,len(Matrix2)])
    ax.set_ylim([0,len(Matrix2)])
    lens = []
    for i in np.unique(kmeans.labels_):
        lens.append(list(kmeans.labels_).count(i))
        leni = sum(lens)
        ax.plot([leni,leni],[-1e3,1e3],'k',lw=1.5)
        ax.plot([-1e3,1e3],[leni,leni],'k',lw=1.5)

allclusters = []
fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10,10),sharex=True,sharey=True)
ClusterAndPlot(ax1,Cmat_eigs[0],4,np.min(Cmat_eigs),np.max(Cmat_eigs),cmap)
ClusterAndPlot(ax2,Cmat_eigs[1],4,np.min(Cmat_eigs),np.max(Cmat_eigs),cmap)
ClusterAndPlot(ax3,Cmat_eigs[2],4,np.min(Cmat_eigs),np.max(Cmat_eigs),cmap)
ClusterAndPlot(ax4,Cmat_eigs[3],4,np.min(Cmat_eigs),np.max(Cmat_eigs),cmap)

axc     = fig.add_axes([0.2, 0.05, 0.6, 0.025])
norm    = mpl.colors.Normalize(vmin=np.min(Cmat_eigs), vmax=np.max(Cmat_eigs))
cb1     = mpl.colorbar.ColorbarBase(axc, cmap=cmap,norm=norm,extend='both',orientation='horizontal')
cb1.set_label('Eigenvector Coefficients',fontsize=15)
axc.tick_params(labelsize=13)
fig.tight_layout

#%% Snapshots

snaps = [265,1555,333,770]

fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10,10),sharex=True,sharey=True)
ax1.pcolormesh(Cmats[snaps[0]],cmap=cmap,vmin=-1,vmax=1)
ax1.set_title('Average '+str(np.round(np.nanmean(Cmats[snaps[0]]),2)),fontsize=16,weight='bold')
ax2.pcolormesh(Cmats[snaps[1]],cmap=cmap,vmin=-1,vmax=1)
ax2.set_title('Average '+str(np.round(np.nanmean(Cmats[snaps[1]]),2)),fontsize=16,weight='bold')
ax3.pcolormesh(Cmats[snaps[2]],cmap=cmap,vmin=-1,vmax=1)
ax3.set_title('Average '+str(np.round(np.nanmean(Cmats[snaps[2]]),2)),fontsize=16,weight='bold')
ax4.pcolormesh(Cmats[snaps[3]],cmap=cmap,vmin=-1,vmax=1)
ax4.set_title('Average '+str(np.round(np.nanmean(Cmats[snaps[3]]),2)),fontsize=16,weight='bold')


axc     = fig.add_axes([0.2, 0.05, 0.6, 0.025])
norm    = mpl.colors.Normalize(vmin=-1, vmax=1)
cb1     = mpl.colorbar.ColorbarBase(axc, cmap=cmap,norm=norm,extend='both',orientation='horizontal')
cb1.set_label('Correlations at various time staps',fontsize=15)
axc.tick_params(labelsize=13)
fig.tight_layout

#%%


allclusters = []
fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10,10),sharex=True,sharey=True)
ax1.set_title('Average '+str(np.round(np.nanmean(Cmats[snaps[0]]),2)),fontsize=16,weight='bold')
ax2.set_title('Average '+str(np.round(np.nanmean(Cmats[snaps[1]]),2)),fontsize=16,weight='bold')
ax3.set_title('Average '+str(np.round(np.nanmean(Cmats[snaps[2]]),2)),fontsize=16,weight='bold')
ax4.set_title('Average '+str(np.round(np.nanmean(Cmats[snaps[3]]),2)),fontsize=16,weight='bold')

ClusterAndPlot(ax1,Cmats[snaps[0]],4,-1,1,cmap)
ClusterAndPlot(ax2,Cmats[snaps[1]],4,-1,1,cmap)
ClusterAndPlot(ax3,Cmats[snaps[2]],4,-1,1,cmap)
ClusterAndPlot(ax4,Cmats[snaps[3]],4,-1,1,cmap)

axc     = fig.add_axes([0.2, 0.05, 0.6, 0.025])
norm    = mpl.colors.Normalize(vmin=-1, vmax=1)
cb1     = mpl.colorbar.ColorbarBase(axc, cmap=cmap,norm=norm,extend='both',orientation='horizontal')
cb1.set_label('Correlations at various time staps',fontsize=15)
axc.tick_params(labelsize=13)
fig.tight_layout