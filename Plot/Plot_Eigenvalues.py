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

#EVs_hip = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/EVs/HIP_hr.pkl')).T[0]
#EVs_pfc = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/EVs/PFC_hr.pkl')).T[0]
#EVs_par = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/EVs/PAR_hr.pkl')).T[0]
EVs_hip = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/EVs/D_HIP.pkl')).T[0]
EVs_pfc = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/EVs/D_PFC.pkl')).T[0]
EVs_par = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/EVs/D_PAR.pkl')).T[0]

# --------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------- #

strs1 = ['(a)','(b)','(c)']
strs = ['Eigenspectrum HIP','Eigenspectrum PFC','Eigenspectrum PAR']
datas = [EVs_hip,EVs_pfc,EVs_par]
colors = ['steelblue','orange','forestgreen']
fig,axes = plt.subplots(3,1,figsize=(12,10),sharex=True,sharey=True)

for i in range(3):
    ax = axes[i]
    ax.plot([-1e3,1e3],[0.02,0.02],'k:',lw=0.5)
    ax.text(-0.2,0.2,strs1[i]+' '+strs[i],fontsize=15)
    ax.set_ylabel('Eigenvalue coefficient')
    ax.plot(datas[i],'o',c=colors[i],label=strs[i])
    ax.plot(datas[i],c=colors[i])
    for j in range(16):
        ax.text(j,datas[i][j]+0.01,str(np.round(datas[i][j]*100,1)),ha='center',
                weight='bold',color=colors[i])
ax.set_xlim([-0.5,15.5])
ax.set_ylim([0,0.23])
ax.set_xlabel('Component')
fig.tight_layout()
plt.savefig('/Users/mmdekker/Documents/Werk/Figures/Neurostuff/Pics_20191210/Eigenspectra_D.png',bbox_inches = 'tight',dpi=200)
