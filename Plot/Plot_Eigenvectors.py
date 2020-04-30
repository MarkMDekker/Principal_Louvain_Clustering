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

EOFs_hip = np.array(pd.read_pickle(path_pdata+'EOFs/HIP/'+ex+'.pkl'))
EOFs_pfc = np.array(pd.read_pickle(path_pdata+'EOFs/PFC/'+ex+'.pkl'))
EOFs_par = np.array(pd.read_pickle(path_pdata+'EOFs/PAR/'+ex+'.pkl'))

EVs_hip = np.array(pd.read_pickle(path_pdata+'EVs/HIP/'+ex+'.pkl')).T[0]
EVs_pfc = np.array(pd.read_pickle(path_pdata+'EVs/PFC/'+ex+'.pkl')).T[0]
EVs_par = np.array(pd.read_pickle(path_pdata+'EVs/PAR/'+ex+'.pkl')).T[0]

# --------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------- #

strs = ['Eigenvectors HIP', 'Eigenvectors PFC', 'Eigenvectors PAR']
datas = [EOFs_hip, EOFs_pfc, EOFs_par]
datas2 = [EVs_hip, EVs_pfc, EVs_par]
colors = ['steelblue', 'orange', 'forestgreen']
fig, axes = plt.subplots(5, 3, figsize=(12, 10), sharex=True, sharey=True)

for i in range(3):
    for j in range(5):
        ax = axes[j][i]
        if j == 0:
            ax.set_title(strs[i])
        ax.plot(np.linspace(1, 99, 50),
                datas[i][j]/np.sign(np.mean(datas[i][j])), 'o')
        ax.text(0.98, 0.9,
                'EOF'+str(j+1)+' ('+str(np.round(datas2[i][j]*100, 1))+'%)',
                transform=ax.transAxes, ha='right')
        if j == 4:
            ax.set_xlabel('Frequency band (Hz)')

fig.tight_layout()
plt.savefig(path_figsave+'Eigenvectors.png', bbox_inches='tight', dpi=200)
