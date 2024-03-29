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

SUBs = ['279418', '279419', '339295', '339296', '340233', '341319',
        '341321', '346110', '346111', '347640', '347641']
REGs = ['HIP', 'PAR', 'PFC']

# ----------------------------------------------------------------- #
# Read data
# ----------------------------------------------------------------- #

config = configparser.ConfigParser()
config.read('../pipeline/config.ini')
Path_Data = config['PATHS']['DATA_RAW']
Path_Datasave = config['PATHS']['DATA_SAVE']
Path_Figsave = ('/Users/mmdekker/Documents/Werk/Figures/Papers/2020_Brain/')
path_eof1 = '/Users/mmdekker/Documents/Werk/Data/Neuro/Processed/EOF1/'
path_eof2 = '/Users/mmdekker/Documents/Werk/Data/Neuro/Processed/EOF2/'
spectra = pd.read_csv('/Users/mmdekker/Documents/Werk/Data/Neuro/Processed/Spectra.csv', index_col=0)
spec_hip = np.array(spectra['HIP'])
spec_par = np.array(spectra['PAR'])
spec_pfc = np.array(spectra['PFC'])
specs_av = [spec_hip, spec_par, spec_pfc]
other = np.array(spectra[spectra.keys()[3:]])
other_hip = other[:, ::3].T
other_par = other[:, 1::3].T
other_pfc = other[:, 2::3].T
others = [other_hip, other_par, other_pfc]

#%%
# --------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------- #

eof1s = [pd.read_csv(path_eof1+REG+'.csv') for REG in REGs]
eof2s = [pd.read_csv(path_eof2+REG+'.csv') for REG in REGs]
eofs = [eof1s, eof2s]
Freqs = np.arange(1, 99+1, 2)
ss = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']
nams = [r'$\delta$', r'$\theta$', r'$\beta$', r'$\gamma$']
titles = ['Hippocampus', 'Parietal cortex', 'Prefrontal cortex']
lows = [0, 5, 12, 32]
upps = [5, 12, 32, 110]
cols = ['steelblue', 'tomato', 'forestgreen', 'purple', 'silver', 'goldenrod']
cols2 = ['dodgerblue', 'forestgreen', 'tomato', 'goldenrod']
sa = [r'$\lambda_1$: ', r'$\lambda_2$: ']
fig, axes2 = plt.subplots(3, 3, figsize=(14, 10))

cmap = plt.cm.coolwarm
for fig_col in range(3):
    REG = REGs[fig_col]
    ax = axes2[0][fig_col]
    ax.set_title(titles[fig_col], fontsize=16)
    ax.text(0.01, 0.98, ss[fig_col],
            fontsize=16, transform=ax.transAxes, va='top')
    if fig_col == 0:
        ax.set_ylabel('Eigenvalue', fontsize=16)
    else:
        ax.set_yticks([])
    ax.set_xticks([])
    ax.plot(range(50), specs_av[fig_col], c='k', lw=2)
    ax.plot(range(50), specs_av[fig_col], 's', c='k', ms=4)
    ax.set_xlim([-1, 25])
    ax.set_ylim([-.002, 0.11])
    ax.text(0.5, 0.02, 'Index', transform=ax.transAxes, ha='center',
            va='bottom', fontsize=16)
    for i in range(2):
        x = i
        y = specs_av[fig_col][i]
        ax.annotate('     '+sa[i]+str(np.round(y*100, 1))+'% of variance', xy=(x, y),
                    xytext=(x+1, y+0.01),
                    arrowprops=dict(facecolor='black', arrowstyle='->'),
                    ha='left', fontsize=12)
        for j in range(len(SUBs)):
            ax.plot(range(50), others[i][j], lw=0.75, zorder=-1,
                    c=cmap(j/len(SUBs)))
    for fig_row in [0, 1]:
        ax = axes2[fig_row+1, fig_col]
        for sub_i in range(len(SUBs)):
            col = cmap(sub_i/len(SUBs))
            ax.semilogx(Freqs, eofs[fig_row][fig_col][SUBs[sub_i]], c=col,
                        lw=0.75, label=SUBs[sub_i])

        ax.plot(Freqs, eofs[fig_row][fig_col]['Average'], 'k', lw=2)
        ax.plot(Freqs, eofs[fig_row][fig_col]['Average'], 's', c='k', ms=4)

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
                            alpha=0.1,
                            interpolate=True, zorder=-1e10)
            ax.text(np.mean([lows[j], upps[j]]), ax.get_ylim()[0]+0.01,
                    nams[j],
                    va='bottom', ha='center', c=cols2[j], fontsize=15)

        # Other ax stuff
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.tick_params(axis='both', which='minor', labelsize=11)
        ax.set_xlim([0.9, 110])
        ax.set_ylim([-0.7, 0.7])
        if fig_row == 1:
            ax.set_xlabel('Frequency (Hz)', fontsize=16)
        else:
            ax.set_xticks([])
        if fig_col == 0:
            ax.set_ylabel('Coefficients', fontsize=16)
        else:
            ax.set_yticks([])
        ax.text(0.01, 0.98, ss[int(3+3*fig_row+fig_col)]+' PC'+str(fig_row+1),
                fontsize=16, transform=ax.transAxes, va='top')
fig.tight_layout()
plt.savefig('/Users/mmdekker/Documents/Werk/Figures/Papers/2020_Brain/Figure_2.png', bbox_inches='tight', dpi=200)
