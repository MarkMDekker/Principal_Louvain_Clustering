# --------------------------------------------------------------------------- #
# General
# --------------------------------------------------------------------------- #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

strings = ['Exploring new','Exploring old','Not exploring']
figdir = '/Users/mmdekker/Documents/Werk/Figures/Neurostuff/Dataset_new/'

ex = '_PAR'
save = 0
Synchronization = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ObjectExperiment/Synchronization'+ex+'.pkl')).T[0]
RelativePower = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ObjectExperiment/RelativePower'+ex+'.pkl'))
Steps = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ObjectExperiment/Steps'+ex+'.pkl')).T[0]
ProcessedPower = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ObjectExperiment/ProcessedPower'+ex+'.pkl'))
#%%
# --------------------------------------------------------------------------- #
# Fourier Transforms (full)
# --------------------------------------------------------------------------- #

fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(12,4),sharey=True)
rs = [rA,rB,rC]
axs = [ax1,ax2,ax3]
for j in range(3):
    ax = axs[j]
    for i in range(len(rs[j])):
        ax.plot(frq,rs[j][i])
    ax.set_title(strings[j])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_xlim([0,50])
    ax.set_ylim([ax.get_ylim()[0],ax.get_ylim()[1]])
    ax.plot([3,3],[-1e3,1e3],'k',lw=1.5)
    ax.plot([8,8],[-1e3,1e3],'k',lw=1.5)
    ax.plot([12,12],[-1e3,1e3],'k',lw=1.5)
    ax.plot([38,38],[-1e3,1e3],'k',lw=1.5)
    ax.plot([42,42],[-1e3,1e3],'k',lw=1.5)
    ax.text(5.5,-1e-4,r'$\theta$',ha='center')
    ax.text(10,-1e-4,r'$\alpha$',ha='center')
    ax.text(25,-1e-4,r'$\beta$',ha='center')
    ax.text(40,-1e-4,r'$\gamma$',ha='center')
ax1.set_ylabel('Power')
fig.tight_layout()
if save==1: plt.savefig(figdir+'FourTrans.png',bbox_inches = 'tight',dpi=200)
plt.show()

#%%
# --------------------------------------------------------------------- #
# Covariance (overall)
# --------------------------------------------------------------------- #

fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(12,4),sharey=True)
cms = [CovMatA,CovMatB,CovMatC]
mn,mx = (np.nanmin(cms),np.nanmax(cms))
axs = [ax1,ax2,ax3]
for j in range(3):
    ax = axs[j]
    ax.pcolormesh(cms[j],vmin=mn,vmax=mx, cmap=plt.cm.rainbow)
    ax.set_title(strings[j])
    ax.set_xlabel('Channel')
ax1.set_ylabel('Channel')
axc     = fig.add_axes([0.24, -0.035, 0.6, 0.03])
norm    = mpl.colors.Normalize(vmin=mn,vmax=mx)
cb1     = mpl.colorbar.ColorbarBase(axc, cmap=plt.cm.rainbow,norm=norm,extend='both',orientation='horizontal')
cb1.set_label('Normalized covariance of signal',fontsize=15)
axc.tick_params(labelsize=13)
fig.tight_layout()
if save==1: plt.savefig(figdir+'CovOverall.png',bbox_inches = 'tight',dpi=200)
plt.show()

#%%
# --------------------------------------------------------------------- #
# Correlation (overall)
# --------------------------------------------------------------------- #

fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(12,4.5),sharey=True)
cms = [CorrMatA,CorrMatB,CorrMatC]
mn,mx = (np.nanmin(cms),np.nanmax(cms))
axs = [ax1,ax2,ax3]
for j in range(3):
    ax = axs[j]
    ax.pcolormesh(cms[j],vmin=mn,vmax=mx, cmap=plt.cm.rainbow)
    ax.set_title(strings[j])
    ax.set_xlabel('Channel')
    ax.set_xlim([ax.get_xlim()[0],ax.get_xlim()[1]])
    ax.set_ylim([ax.get_ylim()[0],ax.get_ylim()[1]])
    ax.plot([16,16],[-1e3,1e3],'k',lw=2,zorder=1e9)
    ax.plot([-1e3,1e3],[16,16],'k',lw=2,zorder=1e9)
    ax.plot([24,24],[-1e3,1e3],'k',lw=2,zorder=1e9)
    ax.plot([-1e3,1e3],[24,24],'k',lw=2,zorder=1e9)
ax1.set_ylabel('Channel')
axc     = fig.add_axes([0.24, -0.035, 0.6, 0.03])
norm    = mpl.colors.Normalize(vmin=mn,vmax=mx)
cb1     = mpl.colorbar.ColorbarBase(axc, cmap=plt.cm.rainbow,norm=norm,extend='both',orientation='horizontal')
cb1.set_label('Correlation of fourier residues',fontsize=15)
axc.tick_params(labelsize=13)
fig.tight_layout()
if save==1: plt.savefig(figdir+'CorrResidu.png',bbox_inches = 'tight',dpi=200)
plt.show()

#%%
# --------------------------------------------------------------------- #
# Plot PCs
# --------------------------------------------------------------------- #
WM = 1
x = 1
y = 2
fig,axes = plt.subplots(4,3,figsize=(12,12),sharey=True)

#Nums = [vals_num,vals_num_x,vals_num_o,vals_num_n]
#pcs = [PCs,PCs_all_x,PCs_all_o,PCs_all_n]
titles = ['(EOFs of) Whole dataset','(EOFs of) Not exploring','(EOFs of) Exploring old object','(EOFs of) Exploring new object']

for i in range(4):
    Num = np.array(vals_nums_spec)[[0,1,2]][i]
    pc = np.array(PCs_all_spec)[[0,1,2]][i]
    ax1,ax2,ax3 = axes[i]
    #ax1.plot(windowmean(pc[x],WM),windowmean(pc[y],WM),'k',lw=0.2)
    ax1.scatter(windowmean(pc[x],WM),windowmean(pc[y],WM),c=plt.cm.coolwarm(Mvec),zorder=5,s=2)
    ax2.scatter(windowmean(pc[x],WM),windowmean(pc[y],WM),c=plt.cm.coolwarm(Newcont[:len(PCs[0])]),zorder=5,s=2+5*Newcont[:len(PCs[0])])
    ax3.scatter(windowmean(pc[x],WM),windowmean(pc[y],WM),c=plt.cm.coolwarm(Oldcont[:len(PCs[0])]),zorder=5,s=2+5*Oldcont[:len(PCs[0])])
    
    ax1.text(0.5,0.95,'Average correlation matrix',transform = ax1.transAxes,ha='center',color='tomato',fontsize=9)
    ax2.text(0.5,0.95,'Exploring new',transform = ax2.transAxes,ha='center',color='tomato',fontsize=9)
    ax3.text(0.5,0.95,'Exploring old',transform = ax3.transAxes,ha='center',color='tomato',fontsize=9)
    ax1.text(0.01,0.5,'PC '+str(y+1)+' ('+str(round(Num[y]*100,2))+'%)',va='center',transform = ax1.transAxes,rotation=90)
    ax1.text(0.5,0.01,'PC '+str(x+1)+' ('+str(round(Num[x]*100,2))+'%)',ha='center', transform = ax1.transAxes)
    ax2.text(0.5,0.01,'PC '+str(x+1)+' ('+str(round(Num[x]*100,2))+'%)',ha='center',transform = ax2.transAxes)
    ax3.text(0.5,0.01,'PC '+str(x+1)+' ('+str(round(Num[x]*100,2))+'%)',ha='center',transform = ax3.transAxes)

    ax2.set_title(titles[i])
fig.tight_layout()
if save==1: plt.savefig(figdir+'PCspace_'+str(x)+str(y)+'.png',bbox_inches = 'tight',dpi=200)
plt.show()

#%%
# --------------------------------------------------------------------- #
# Compare EOFs
# --------------------------------------------------------------------- #

fig,axes = plt.subplots(3,1,figsize=(12,12),sharex=True)
string = ['Whole','Not exploring','Not exploring','Exploring new','Exploring new','Exploring old','Exploring old']
cols = ['k','silver','silver','steelblue','steelblue','orange','orange']
for i in range(len(EOFs_spec)):
    ax1,ax2,ax3 = axes
    ax1.plot(EOFs_spec[i][0],'o',label=string[i],c=cols[i])
    ax2.plot(EOFs_spec[i][1],'o',c=cols[i])
    ax3.plot(EOFs_spec[i][2],'o',c=cols[i])
ax1.legend(loc='best')
ax1.set_ylabel('EOF coefficient')
ax2.set_ylabel('EOF coefficient')
ax3.set_ylabel('EOF coefficient')
ax1.set_xlabel('Index of time series')
fig.tight_layout()
if save==1: plt.savefig(figdir+'EOFs.png',bbox_inches = 'tight',dpi=200)

#%%
# --------------------------------------------------------------------- #
# Investigate relation between 
# --------------------------------------------------------------------- #

Syn = Mvec
N = Newcont[:len(Mvec)]
O = Oldcont[:len(Mvec)]
CorrO = np.corrcoef(Syn[~np.isnan(O)],O[~np.isnan(O)])[0,1]
CorrN = np.corrcoef(Syn[~np.isnan(N)],N[~np.isnan(N)])[0,1]

fig,(ax1,ax2) = plt.subplots(2,1,figsize=(12,12),sharex=True)
ax1.plot(Mvec,O,'o',ms=1)
ax2.plot(Mvec,N,'o',ms=1)
ax1.set_title('Old exploration: '+str(np.round(CorrO,2)))
ax2.set_title('New exploration: '+str(np.round(CorrN,2)))
