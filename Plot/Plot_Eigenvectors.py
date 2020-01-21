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

#EOFs_hip = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/EOFs/HIP_hr.pkl'))
#EOFs_pfc = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/EOFs/PFC_hr.pkl'))
#EOFs_par = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/EOFs/PAR_hr.pkl'))
EOFs_hip = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/EOFs/D_HIP.pkl'))
EOFs_pfc = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/EOFs/D_PFC.pkl'))
EOFs_par = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/EOFs/D_PAR.pkl'))

#EVs_hip = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/EVs/HIP_hr.pkl')).T[0]
#EVs_pfc = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/EVs/PFC_hr.pkl')).T[0]
#EVs_par = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/EVs/PAR_hr.pkl')).T[0]
EVs_hip = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/EVs/D_HIP.pkl')).T[0]
EVs_pfc = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/EVs/D_PFC.pkl')).T[0]
EVs_par = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/EVs/D_PAR.pkl')).T[0]

#%%
# --------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------- #

#strs1 = ['(a)','(b)','(c)']
strs = ['Eigenvectors HIP','Eigenvectors PFC','Eigenvectors PAR']
datas = [EOFs_hip,EOFs_pfc,EOFs_par]
datas2 = [EVs_hip,EVs_pfc,EVs_par]
colors = ['steelblue','orange','forestgreen']
fig,axes = plt.subplots(5,3,figsize=(12,10),sharex=True,sharey=True)

for i in range(3):
    for j in range(5):
        ax = axes[j][i]
        if j==0: ax.set_title(strs[i])
        ax.plot(datas[i][j]/np.sign(np.mean(datas[i][j])),'o')
        ax.text(0.02,0.9,'EOF'+str(j+1)+' ('+str(np.round(datas2[i][j]*100,1))+'%)',transform=ax.transAxes)


#for i in range(3):
#    ax = axes[i]
#    ax.plot([-1e3,1e3],[0.02,0.02],'k:',lw=0.5)
#    ax.text(-0.2,0.2,strs1[i]+' '+strs[i],fontsize=15)
#    ax.set_ylabel('Eigenvalue coefficient')
#    ax.plot(datas[i],'o',c=colors[i],label=strs[i])
#    ax.plot(datas[i],c=colors[i])
#    for j in range(16):
#        ax.text(j,datas[i][j]+0.01,str(np.round(datas[i][j]*100,1)),ha='center',
#                weight='bold',color=colors[i])
#ax.set_xlim([-0.5,15.5])
#ax.set_ylim([0,0.23])
#ax.set_xlabel('Component')
fig.tight_layout()
plt.savefig('/Users/mmdekker/Documents/Werk/Figures/Neurostuff/Pics_20191210/Eigenvectors_D.png',bbox_inches = 'tight',dpi=200)
