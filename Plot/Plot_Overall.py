# --------------------------------------------------------------------------- #
# Script to create figure that explores the data in general with synchronicity
# and the normalized relative power
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------- #
# Preambules and paths
# --------------------------------------------------------------------- #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io

strings = ['Exploring new','Exploring old','Not exploring']
figdir = '/Users/mmdekker/Documents/Werk/Figures/Neurostuff/Dataset_new/'
datadir = '/Users/mmdekker/Documents/Werk/Data/Braindata/ObjectTest.mat'

# --------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------- #

ex = 'PFC_hr'
save = 1

# --------------------------------------------------------------------- #
# Read data
# --------------------------------------------------------------------- #

Synchronization = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/Synchronization/'+ex+'.pkl')).T[0]
RelativePower = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/RelativePower/'+ex+'.pkl'))
Steps = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/Steps/'+ex+'.pkl')).T[0]
ProcessedPower = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/ProcessedPower/'+ex+'.pkl'))
if 'NotExploring' not in locals():
    mat = scipy.io.loadmat(datadir)
    DataAll = mat['EEG'][0][0]
    Time        = DataAll[14]
    Data_dat    = DataAll[15]
    Data_re1    = DataAll[17]
    Data_re2    = DataAll[18]
    Data_re3    = DataAll[19]
    Data_re4    = DataAll[25]
    Data_re5    = DataAll[30]
    Data_re6    = DataAll[39]
    Data_vid    = DataAll[40].T
    ExploringNew = np.where(Data_vid[2] == 1)[0]
    ExploringOld = np.where(Data_vid[3] == 1)[0]
    NotExploring = np.array(list(np.where(Data_vid[2] != 1)[0]) + list(np.where(Data_vid[3] != 1)[0]))

# --------------------------------------------------------------------- #
# Relative power and synchronization
# --------------------------------------------------------------------- #

colors = []
for i in range(len(Steps)):
    st = Steps[i]
    if st in ExploringNew:
        colors.append('steelblue')
    elif st in ExploringOld:
        colors.append('orange')
    else:
        colors.append('silver')
#%%
fig,(ax1,ax2) = plt.subplots(2,1,figsize=(15,10),sharex=True)
ax1.scatter(Steps/1000.,Synchronization,c=colors,s=2)
ax1.scatter(Steps[0]/1000.,Synchronization[0],c='steelblue',s=2,zorder=-1,label='Exploring New')
ax1.scatter(Steps[0]/1000.,Synchronization[0],c='orange',s=2,zorder=-1,label='Exploring Old')
ax1.set_ylabel('Average of correlation matrix')
c=ax2.pcolormesh(Steps/1000.,np.arange(0,100,2),ProcessedPower,vmin=-1,vmax=1,
               cmap = plt.cm.coolwarm)
#c=ax2.pcolormesh(Steps/1000.,range(100),Power_proc,vmin=0,vmax=0.002,
#               cmap = plt.cm.rainbow)

# Borders of PC periods
#cols = ['silver','silver','steelblue','steelblue','orange','orange']
#mins = [133,400,70,475,195,315]
#maxs = [165,425,80,491,217,337]
cols = ['steelblue','orange','grey']
mins = [70,83,90]
maxs = [75,89,99]

#ax1.set_ylim([ax1.get_ylim()[0],ax1.get_ylim()[1]])
ax1.set_ylim([0.3,0.9])
ax2.set_ylim([ax2.get_ylim()[0],ax2.get_ylim()[1]])
#for i in range(len(cols)):
#    x = np.arange(mins[i],maxs[i]+1)
#    y1 = np.zeros(len(x))+1e3
#    y2 = np.zeros(len(x))-1e3
#    ax1.fill_between(x,y1,y2,where=y1>y2,interpolate='nearest',alpha=0.25,color=cols[i],zorder=-10)
#    ax1.plot([mins[i],mins[i]],[-1e3,1e3],'k',lw=0.5)
#    ax1.plot([maxs[i],maxs[i]],[-1e3,1e3],'k',lw=0.5)
#    ax2.plot([mins[i],mins[i]],[-1e3,1e3],'k',lw=4)
#    ax2.plot([maxs[i],maxs[i]],[-1e3,1e3],'k',lw=4)

ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Frequency (Hz)')
ax1.set_xlim([0,ax1.get_xlim()[1]])
ax2.set_xlim([ax1.get_xlim()[0],500])#ax1.get_xlim()[1]])
#ax2.plot(ExploringNew/1000.,np.zeros(len(ExploringNew))-1,'o',c='steelblue',ms=3,label='Exploring new object')
#ax2.plot(ExploringOld/1000.,np.zeros(len(ExploringOld))-1,'o',c='orange',ms=3,label='Exploring old object')
ax1.legend(loc='best')
plt.colorbar(c,orientation='horizontal',label='Normalized relative power')
fig.tight_layout
if save==1: plt.savefig('/Users/mmdekker/Documents/Werk/Figures/Neurostuff/Pics_11112019/PowerSynchr_'+ex+'.png',bbox_inches = 'tight',dpi=200)
plt.show()