# --------------------------------------------------------------------- #
# Preambule
# --------------------------------------------------------------------- #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import sys

# --------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------- #

save = 1
ex = 'PFC_hr1'
WM = 1
x = 1
y = 2
dirry = '/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/'

# --------------------------------------------------------------------- #
# Read some extra data
# --------------------------------------------------------------------- #

Newcont = np.array(pd.read_pickle(dirry+'NewOld/NewCont_hr.pkl')).T[0]
Oldcont = np.array(pd.read_pickle(dirry+'NewOld/OldCont_hr.pkl')).T[0]
PCs_all_spec=pd.read_pickle(dirry+'/PCs/'+ex+'.pkl')
vals_nums_spec=pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/EVs/'+ex+'.pkl')
Synchronization = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/Synchronization/'+ex+'.pkl')).T[0]
RelativePower = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/RelativePower/'+ex+'.pkl'))
Steps = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/Steps/'+ex+'.pkl')).T[0]
ProcessedPower = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/ProcessedPower/'+ex+'.pkl'))

Mvec = np.copy(Synchronization)
Mvec = Mvec - np.min(Mvec)
Mvec = Mvec / np.max(Mvec)

# --------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------- #

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
    
def CreateHist(X,Y):
    maxX = np.nanmax(X)
    maxY = np.nanmax(Y)
    minX = np.nanmin(X)
    minY = np.nanmin(Y)
    inds = np.arange(len(X))
    
    arrX = np.linspace(minX,maxX,50)
    arrY = np.linspace(minY,maxY,50)
    hist = np.zeros(shape=(50,50))
    for i in range(len(hist)-1):
        for j in range(len(hist[0])-1):
            indsX = inds[(X>=arrX[i]) & (X<arrX[i+1])]
            indsY = inds[(Y>=arrY[j]) & (Y<arrY[j+1])]
            hist[i,j] = len(np.intersect1d(indsX,indsY))
    return arrX,arrY,hist

#%%
# --------------------------------------------------------------------- #
# Plot PCs
# --------------------------------------------------------------------- #

fig,axes = plt.subplots(4,4,figsize=(12,12),sharey=True)

#Nums = [vals_num,vals_num_x,vals_num_o,vals_num_n]
#pcs = [PCs,PCs_all_x,PCs_all_o,PCs_all_n]
titles = ['(EOFs of) Whole dataset','(EOFs of) Not exploring','(EOFs of) Exploring old object','(EOFs of) Exploring new object']

for i in range(4):
    Num = np.array(vals_nums_spec)[[0,3,2,1]][i]
    pc = np.array(PCs_all_spec)[[0,3,2,1]][i]
    ax1,ax2,ax3,ax4 = axes[i]
    #ax1.plot(windowmean(pc[x],WM),windowmean(pc[y],WM),'k',lw=0.2)
    ax1.scatter(windowmean(pc[x],WM),windowmean(pc[y],WM),c=plt.cm.coolwarm(Mvec),zorder=5,s=0.1)
    ax2.scatter(windowmean(pc[x],WM),windowmean(pc[y],WM),c=plt.cm.coolwarm(Newcont[:len(pc[0])]),zorder=5,s=0.1)
    ax3.scatter(windowmean(pc[x],WM),windowmean(pc[y],WM),c=plt.cm.coolwarm(Oldcont[:len(pc[0])]),zorder=5,s=0.1)
    vecX,vecY,vecZ = CreateHist(pc[x],pc[y])
    ax4.contourf(vecX,vecY,vecZ.T,np.arange(0,np.nanmax(vecZ)+1,1),cmap=plt.cm.gist_heat_r)
    
    ax1.scatter(windowmean(pc[x],WM),windowmean(pc[y],WM),c=plt.cm.coolwarm(Mvec),zorder=5,s=5*np.sign(Mvec-0.5)/2+0.5)
    ax2.scatter(windowmean(pc[x],WM),windowmean(pc[y],WM),c=plt.cm.coolwarm(Newcont[:len(pc[0])]),zorder=5,s=5*Newcont[:len(pc[0])])
    ax3.scatter(windowmean(pc[x],WM),windowmean(pc[y],WM),c=plt.cm.coolwarm(Oldcont[:len(pc[0])]),zorder=5,s=5*Oldcont[:len(pc[0])])
    ax4.contourf(vecX,vecY,vecZ.T,np.arange(0,np.nanmax(vecZ)+1,1),cmap=plt.cm.gist_heat_r)
    
    ax1.text(0.5,0.95,'Average correlation matrix',transform = ax1.transAxes,ha='center',color='tomato',fontsize=9)
    ax2.text(0.5,0.95,'Exploring new',transform = ax2.transAxes,ha='center',color='tomato',fontsize=9)
    ax3.text(0.5,0.95,'Exploring old',transform = ax3.transAxes,ha='center',color='tomato',fontsize=9)
    ax4.text(0.5,0.95,'PDF',transform = ax4.transAxes,ha='center',color='tomato',fontsize=9)
    ax1.text(0.01,0.5,'PC '+str(y+1)+' ('+str(round(Num[y]*100,2))+'%)',va='center',transform = ax1.transAxes,rotation=90)
    ax1.text(0.5,0.01,'PC '+str(x+1)+' ('+str(round(Num[x]*100,2))+'%)',ha='center', transform = ax1.transAxes)
    ax2.text(0.5,0.01,'PC '+str(x+1)+' ('+str(round(Num[x]*100,2))+'%)',ha='center',transform = ax2.transAxes)
    ax3.text(0.5,0.01,'PC '+str(x+1)+' ('+str(round(Num[x]*100,2))+'%)',ha='center',transform = ax3.transAxes)
    ax4.text(0.5,0.01,'PC '+str(x+1)+' ('+str(round(Num[x]*100,2))+'%)',ha='center',transform = ax4.transAxes)

    ax2.set_title(titles[i])
fig.tight_layout()
if save==1: plt.savefig('/Users/mmdekker/Documents/Werk/Figures/Neurostuff/Pics_04112019/PCspace_'+str(x)+str(y)+'_'+ex+'.png',bbox_inches = 'tight',dpi=200)
plt.show()