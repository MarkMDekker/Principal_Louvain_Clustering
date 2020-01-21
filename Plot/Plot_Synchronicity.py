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

save = 1

# --------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------- #

S_hip = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/Synchronization/HIP_hr.pkl')).T[0]
S_pfc = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/Synchronization/PFC_hr.pkl')).T[0]
S_par = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/Synchronization/PAR_hr.pkl')).T[0]

## --------------------------------------------------------------------- #
## Select data
## --------------------------------------------------------------------- #

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

ExploringNew = ExploringNew[ExploringNew<499000]
ExploringOld = ExploringOld[ExploringOld<499000]
NotExploring = NotExploring[NotExploring<499000]

#%%
# --------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------- #

datas = [S_hip,S_pfc,S_par]
labs = ['HIP','PFC','PAR']
#strs1 = ['(a)','(b)','(c)']
#strs = ['Eigenspectrum HIP','Eigenspectrum PFC','Eigenspectrum PAR']
#colors = ['steelblue','orange','forestgreen']
fig,axes = plt.subplots(3,3,figsize=(20,20),sharex=True,sharey=True)

for i in range(3):
    for j in range(3):
        ax = axes[j][i]
        if i!=j:
            ax.scatter(datas[i][NotExploring],datas[j][NotExploring],s=1,c='silver',label='Not exploring')
            ax.scatter(datas[i][ExploringOld],datas[j][ExploringOld],s=1,c='forestgreen',label='Exploring old')
            ax.scatter(datas[i][ExploringNew],datas[j][ExploringNew],s=1,c='tomato',label='Exploring new') 

axes[0][1].legend()
for i in range(3):
    axes[-1][i].set_xlabel(labs[i])
    axes[i][0].set_ylabel(labs[i])
#ax.set_xlim([-0.5,15.5])
#ax.set_ylim([0,0.23])
#ax.set_xlabel('Component')
fig.tight_layout()
plt.savefig('/Users/mmdekker/Documents/Werk/Figures/Neurostuff/Pics_11112019/SynchSpace.png',bbox_inches = 'tight',dpi=200)
