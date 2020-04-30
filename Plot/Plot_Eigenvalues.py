# --------------------------------------------------------------------- #
# Preambules and paths
# --------------------------------------------------------------------- #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
path_pdata = ('/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/'
              'ProcessedData/')
path_figsave = ('/Users/mmdekker/Documents/Werk/Figures/Figures_Side/Brain/'
                'Pics_newest/')

# --------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------- #

ex = '600000_0'

# --------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------- #

EVs_hip = np.array(pd.read_pickle(path_pdata+'EVs/HIP/'+ex+'.pkl')).T[0]
EVs_pfc = np.array(pd.read_pickle(path_pdata+'EVs/PFC/'+ex+'.pkl')).T[0]
EVs_par = np.array(pd.read_pickle(path_pdata+'EVs/PAR/'+ex+'.pkl')).T[0]

# --------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------- #

strs1 = ['(a)', '(b)', '(c)']
strs = ['Eigenspectrum HIP', 'Eigenspectrum PFC', 'Eigenspectrum PAR']
datas = [EVs_hip, EVs_pfc, EVs_par]
colors = ['steelblue', 'orange', 'forestgreen']
fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, sharey=True)

for i in range(3):
    ax = axes[i]
    ax.plot([-1e3,1e3],[0.02,0.02],'k:',lw=0.5)
    ax.text(0.01, 0.98, strs1[i]+' '+strs[i],fontsize=15,
            transform=ax.transAxes, va='top')
    ax.set_ylabel('Eigenvalue coefficient')
    ax.plot(datas[i],'o',c=colors[i],label=strs[i])
    ax.plot(datas[i],c=colors[i])
    for j in range(16):
        ax.text(j,datas[i][j]+0.01,str(np.round(datas[i][j]*100,1)),ha='center',
                weight='bold',color=colors[i])
ax.set_xlim([-0.5,15.5])
ax.set_ylim([0,0.15])
ax.set_xlabel('Component')
fig.tight_layout()
plt.savefig(path_figsave+'Eigenvalues.png', bbox_inches='tight', dpi=200)
