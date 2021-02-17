# ----------------------------------------------------------------- #
# About script
# Figure 2: eigenvectors
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import configparser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------- #
# Input
# ----------------------------------------------------------------- #

SUBs = ['279418', '279419', '339295', '339296']#, '340233', '341319',
        #'341321', '346110', '346111', '347640', '347641']
REGs = ['HIP', 'PAR', 'PFC']

# ----------------------------------------------------------------- #
# Read data
# ----------------------------------------------------------------- #

config = configparser.ConfigParser()
config.read('../pipeline/config.ini')
Path_Data = config['PATHS']['DATA_RAW']
Path_Datasave = config['PATHS']['DATA_SAVE']
Path_Figsave = ('/Users/mmdekker/Documents/Werk/Figures/Papers/2020_Brain/')
path_eof1 = '/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/Processed/EOF1/'
path_eof2 = '/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/Processed/EOF2/'

# --------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------- #

eof1s = [pd.read_csv(path_eof1+REG+'.csv') for REG in REGs]
eof2s = [pd.read_csv(path_eof2+REG+'.csv') for REG in REGs]
eofs = [eof1s, eof2s]
Freqs = np.arange(1, 99+1, 2)
ss = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
nams = [r'$\delta$', r'$\theta$', r'$\beta$', r'$\gamma$']
titles = ['Hippocampus', 'Parietal cortex', 'Prefrontal cortex']
lows = [0, 5, 12, 32]
upps = [5, 12, 32, 100]
cols = ['steelblue', 'tomato', 'forestgreen', 'purple', 'silver', 'goldenrod']
cols2 = ['dodgerblue', 'forestgreen', 'tomato', 'goldenrod']
fig, axes2 = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)

for fig_col in range(3):
    REG = REGs[fig_col]
    for fig_row in range(2):
        ax = axes2[fig_row, fig_col]
        
        for sub_i in range(len(SUBs)):
            col = plt.cm.Spectral_r(sub_i/len(SUBs))
            #if SUBs[sub_i] == '346111':
            ax.plot(Freqs, eofs[fig_row][fig_col][SUBs[sub_i]], c=col, lw=0.75, label=SUBs[sub_i])
            
        ax.plot(Freqs, eofs[fig_row][fig_col]['Average'], 'k', lw=1)
        ax.plot(Freqs, eofs[fig_row][fig_col]['Average'], 's', c='k', ms=3)

        # Spectrum parts
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
                            alpha=0.05,
                            interpolate=True, zorder=-1e10)
            ax.text(np.mean([lows[j], upps[j]]), ax.get_ylim()[0]+0.01,
                    nams[j],
                    va='bottom', ha='center', c=cols2[j], fontsize=15)

        # Other ax stuff
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        if fig_row == 1:
            ax.set_xlabel('Frequency (Hz)', fontsize=14)
        else:
            ax.set_title(titles[fig_col])
        if fig_col == 0:
            ax.set_ylabel('Coefficients', fontsize=14)
        ax.text(0.01, 0.98, ss[int(3*fig_row+fig_col)]+' EOF '+str(fig_row+1),
                fontsize=12, transform=ax.transAxes, va='top')
fig.tight_layout()
#plt.savefig(Path_Figsave+'Figure_2.png', bbox_inches='tight', dpi=200)
