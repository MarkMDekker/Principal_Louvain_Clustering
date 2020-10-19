# ----------------------------------------------------------------- #
# About script
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import configparser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------- #
# Input
# ----------------------------------------------------------------- #

#Subjects: 279418, 279419, 339295

N = -1
PRT = 0
SUBs = ['279418', '279419', '339295', '339296', '346110', '346111']
res = 200
tau = 30
REGs = ['HIP', 'PAR', 'PFC']

# ----------------------------------------------------------------- #
# Read data
# ----------------------------------------------------------------- #

config = configparser.ConfigParser()
config.read('../config.ini')
Path_Data = config['PATHS']['DATA_RAW']
Path_Datasave = config['PATHS']['DATA_SAVE']
Path_Figsave = config['PATHS']['FIG_SAVE']

# --------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------- #

ss = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']
nams = [r'$\delta$', r'$\theta$', r'$\beta$', r'$\gamma$']
lows = [0, 5, 12, 32]
upps = [5, 12, 32, 100]
cols = ['steelblue', 'tomato', 'forestgreen', 'purple', 'silver', 'goldenrod']
cols2 = ['dodgerblue', 'forestgreen', 'tomato', 'goldenrod']
fig, axes2 = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)

for k in range(3):
    axes = axes2[k]
    REG = REGs[k]
    EOFss = []
    EVss = []
    for i in range(len(SUBs)):
        Name = 'F_'+REG+'_'+SUBs[i]
        EOFs = np.array(pd.read_pickle(Path_Datasave+'EOFs/'+Name+'.pkl'))[:25]
        EVs = np.array(pd.read_pickle(Path_Datasave+'Eigenvalues/'+Name +
                                      '.pkl')).T[0]
        EOFss.append(EOFs)
        EVss.append(EVs)
    Freqs = np.arange(1, 99+1, 2)
    EOFss = np.array(EOFss)
    for i in range(len(EOFss)):
        Averages = EOFss[:i].mean(axis=0)
        for j in range(len(EOFss[0])):
            r = np.corrcoef(Averages[j], EOFss[i][j])[0][1]
            if r < 0:
                EOFss[i][j] = -EOFss[i][j]
        Averages = EOFss[:i].mean(axis=0)
    Averages = EOFss.mean(axis=0)
    for i in range(3):
        ax = axes[i]
        ran = Freqs
        ax.plot(ran, Averages[i], c='k', lw=2, zorder=1e9, label='Average')
        for j in range(len(SUBs)):
            ax.plot(ran, EOFss[j][i], c=cols[j], zorder=-1, lw=0.6)
            ax.plot(ran, EOFss[j][i], 'o', ms=3, c=cols[j], zorder=-1,
                    label=SUBs[j]+': '+str(np.round(EVss[j][i]*100, 1))+'%')
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        ax.text(0.01, 0.98, ss[int(3*k+i)]+' '+REGs[k]+', EOF '+str(i+1),
                fontsize=12, transform=ax.transAxes, va='top')
        ax.set_ylim([-0.6, 0.6])
        ax.set_xlim([0, np.max(Freqs)+1])
        ax.plot([-1e3, 1e3], [0, 0], 'k', lw=0.5)
        for j in list(lows)+list(upps):
            ax.plot([j, j], [-1e3, 1e3], 'k', lw=0.5)
        for j in range(len(nams)):
            x = np.arange(lows[j], upps[j]+0.1, 1)
            y1 = [-1e3]*len(x)
            y2 = [1e3]*len(x)
            ax.fill_between(x, y1, y2, y2 >= y1, facecolor=cols2[j],
                            alpha=0.12,
                            interpolate=True, zorder=-1e10)
            ax.text(np.mean([lows[j], upps[j]]), ax.get_ylim()[0]+0.01,
                    nams[j],
                    va='bottom', ha='center', c=cols2[j], fontsize=15)
        ax.legend(loc='lower right', fontsize=6.5)  # , ncol=2)
        if k == 2:
            ax.set_xlabel('Frequency (Hz)', fontsize=14)
        if i == 0:
            ax.set_ylabel('Coefficients', fontsize=14)
    axes[-1].set_xticks(lows+upps)
fig.tight_layout()
#plt.savefig(Path_Figsave+'EigVec/Eigenvectors.png', bbox_inches='tight', dpi=200)
