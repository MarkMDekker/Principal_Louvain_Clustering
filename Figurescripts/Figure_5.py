# ----------------------------------------------------------------- #
# About script
# Figure 5: exploration biases across clusters
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import pandas as pd
import scipy.stats
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
Path_Figsave = ('/Users/mmdekker/Documents/Werk/Figures/Papers/2020_Brain/')
savepath = '/Users/mmdekker/Documents/Werk/Data/Neuro/Processed/'

# ----------------------------------------------------------------- #
# Input
# ----------------------------------------------------------------- #

RES = 10
TAU = 30
space = np.linspace(-12, 12, RES)

# ----------------------------------------------------------------- #
# Read data
# ----------------------------------------------------------------- #

# With smoothening
DF_meta_s = pd.read_csv(savepath+'ClusterAnalysis_s.csv')
biases_s = np.array([DF_meta_s.Explbias_r1,
                     DF_meta_s.Explbias_r2,
                     DF_meta_s.Explbias_r3,
                     DF_meta_s.Explbias_r4,
                     DF_meta_s.Explbias_r5,
                     DF_meta_s.Explbias_r6,
                     DF_meta_s.Explbias_r7,
                     DF_meta_s.Explbias_r8,
                     DF_meta_s.Explbias_r9,
                     DF_meta_s.Explbias_r10,
                     DF_meta_s.Explbias_r11])
avbias_s = np.array(DF_meta_s.Explbias)

amounts_s = []
for j in range(len(DF_meta_s)):
    amount = np.array([DF_meta_s.Perc_r1[j],
                       DF_meta_s.Perc_r2[j],
                       DF_meta_s.Perc_r3[j],
                       DF_meta_s.Perc_r4[j],
                       DF_meta_s.Perc_r5[j],
                       DF_meta_s.Perc_r6[j],
                       DF_meta_s.Perc_r7[j],
                       DF_meta_s.Perc_r8[j],
                       DF_meta_s.Perc_r9[j],
                       DF_meta_s.Perc_r10[j],
                       DF_meta_s.Perc_r11[j]])
    amounts_s.append(amount)
amounts_s = np.array(amounts_s).T

# Filter out near-empty ones:
biases2_s = np.copy(biases_s)
for i in range(len(amounts_s[0])):
    mean = np.mean(amounts_s[:, i])
    std = np.std(amounts_s[:, i])
    for j in range(len(amounts_s)):
        if amounts_s[j, i] < 0.02:
            biases2_s[j, i] = np.nan
# idx_s1 = np.where(DF_meta_s['AmountData'] > 1000)[0]
# idx_s2 = np.argsort(avbias_s)
# idx_s = []
# for i in range(len(idx_s2)):
#     if idx_s2[i] in idx_s1:
#         idx_s.append(idx_s2[i])
# idx_s = np.array(idx_s)
idx_s = np.argsort(avbias_s)

# With shuffling
DF_meta_sh = pd.read_csv(savepath+'ClusterAnalysis_sh.csv')
biases_sh = np.array([DF_meta_sh.Explbias_r1,
                      DF_meta_sh.Explbias_r2,
                      DF_meta_sh.Explbias_r3,
                      DF_meta_sh.Explbias_r4,
                      DF_meta_sh.Explbias_r5,
                      DF_meta_sh.Explbias_r6,
                      DF_meta_sh.Explbias_r7,
                      DF_meta_sh.Explbias_r8,
                      DF_meta_sh.Explbias_r9,
                      DF_meta_sh.Explbias_r10,
                      DF_meta_sh.Explbias_r11])
avbias_sh = np.array(DF_meta_sh.Explbias)

amounts_sh = []
for j in range(len(DF_meta_sh)):
    amount = np.array([DF_meta_sh.Perc_r1[j],
                       DF_meta_sh.Perc_r2[j],
                       DF_meta_sh.Perc_r3[j],
                       DF_meta_sh.Perc_r4[j],
                       DF_meta_sh.Perc_r5[j],
                       DF_meta_sh.Perc_r6[j],
                       DF_meta_sh.Perc_r7[j],
                       DF_meta_sh.Perc_r8[j],
                       DF_meta_sh.Perc_r9[j],
                       DF_meta_sh.Perc_r10[j],
                       DF_meta_sh.Perc_r11[j]])
    amounts_sh.append(amount)
amounts_sh = np.array(amounts_sh).T

# Filter out near-empty ones:
biases2_sh = np.copy(biases_sh)
for i in range(len(amounts_sh[0])):
    # mean = np.mean(amounts_sh[:, i])
    # std = np.std(amounts_sh[:, i])
    for j in range(len(amounts_sh)):
        if amounts_sh[j, i] < 0.02:
            biases2_sh[j, i] = np.nan
idx_sh = np.argsort(avbias_sh)

Mean = np.mean(pd.read_csv(savepath+'MC_means.csv')['0'])
STD = np.mean(pd.read_csv(savepath+'MC_stds.csv')['0'])

#%%   
# ----------------------------------------------------------------- #
# Plot biases
# ----------------------------------------------------------------- #


def renormalize(vec):
    vec = vec - 0.5
    vec[vec < 0.] = 0
    vec[vec > 1.] = 1
    return vec


FS = 18
DFs = [DF_meta_s, DF_meta_sh]
bias = [biases2_s, biases2_sh]
biasav = [avbias_s, avbias_sh]
AM = [amounts_s, amounts_sh]
idxs = [idx_s, idx_sh]
whs = [np.where(np.array(DFs[0].WilcoxP)[idxs[0]] < 0.2)[0],
       np.where(np.array(DFs[1].WilcoxP)[idxs[1]] < 0.2)[0]]
cmap1 = plt.cm.coolwarm
cols = ['steelblue', 'tomato', 'forestgreen']
fig, ((ax1, ax4), (ax2, ax5), (ax3, ax6)) = plt.subplots(3, 2, figsize=(20, 13))

# Bias plots
ax1.text(0.02, 0.98, '(a)', fontsize=FS,
         transform=ax1.transAxes, va='top')
ax4.text(0.02, 0.98, '(b)',
         fontsize=FS, transform=ax4.transAxes, va='top')
for i in range(2):
    ax = [ax1, ax4][i]
    DF = DFs[i]
    ix = idxs[i]
    bi = bias[i]
    wh = whs[i]
   # ax.plot(np.arange(len(DF)), DF.Explbias[ix], c='k',
   #         label='Exploring', zorder=-1, lw=0.5)
    ax.scatter(range(len(ix)), DF.Explbias[ix],
               c='k', s=15)
    ax.scatter(np.arange(len(ix))[wh], DF.Explbias[ix[wh]],
               c=renormalize(biasav[i][ix[wh]]), vmin=0, vmax=1,
               cmap=cmap1, edgecolor='k', s=200, lw=2)
    ax.plot([-1e3, 1e3], [Mean, Mean], 'k', lw=0.25)
    ax.plot([-1e3, 1e3], [Mean+2*STD, Mean+2*STD], 'k', lw=0.25)
    ax.plot([-1e3, 1e3], [Mean-2*STD, Mean-2*STD], 'k', lw=0.25)
    x = np.arange(-1e3, 1e3)
    y1 = np.array([Mean+2*STD]*len(x))
    y0 = np.array([Mean-2*STD]*len(x))
    ax.fill_between(x, y1, y0, where=y1>=y0, alpha=0.8, color='silver', zorder=-1e9999)
    for j in range(len(bi)):
        #wha = np.where(am[j] > 0.00167)[0]
        ax.scatter(range(len(ix)), bi[j][ix], 
                   c='silver', s=4, zorder=-1,
                   marker='s')
        ax.scatter(np.arange(len(ix))[wh], bi[j][ix[wh]],
                   c=renormalize(bi[j][ix[wh]]), cmap=cmap1, s=30, zorder=-1,
                   marker='s', edgecolor='k', lw=0.5, vmin=0, vmax=1)
    #ax.set_yscale('log')
    ax.plot([-1e3, 1e3], [1, 1], 'k:', lw=1, zorder=-1)
    if i == 0:
        ax.text(len(ix)-0.5, 1, r'$\rightarrow$ exploring', rotation=90, va='bottom', fontsize=FS-8, c='tomato')
        ax.text(len(ix)-0.5, 1, r'not exploring $\leftarrow$ ', rotation=90, va='top', fontsize=FS-8, c='steelblue')
    for j in range(len(wh)):
        ax.plot([wh[j], wh[j]], [-1, np.array(DF.Explbias[ix[wh]])[j]], c=cmap1(renormalize(biasav[i][ix[wh]])[j]), lw=0.5, zorder=-10)
    ax.set_yscale('symlog', linscaley=1)
    ax.set_ylim([-0.1, 8])

# Amount data
stringy = ['(c)', '(d)']
for i in range(2):
    ax = [ax2, ax5][i]
    DF = DFs[i]
    ix = idxs[i]
    bi = bias[i]
    wh = whs[i]
    ax.text(0.02, 0.98, stringy[i], fontsize=FS,
            transform=ax.transAxes, va='top')
    ax.scatter(range(len(ix)), DF.AmountData[ix],
                c='k', s=25)
    ax.scatter(np.arange(len(ix))[wh], DF.AmountData[ix[wh]],
                c=renormalize(biasav[i][ix[wh]]), vmin=0, vmax=1,
                cmap=cmap1, edgecolor='k', s=200, lw=2)
    ax.semilogy(range(len(ix)), DF.AmountData[ix], 'k', lw=0., zorder=-1)
    ax.set_xlim([ax.get_xlim()[0], ax.get_xlim()[1]])
    ax.set_ylim([ax.get_ylim()[0], ax.get_ylim()[1]])
    ax.set_ylim([400, 2000000])
    for j in range(10):
        ax.plot([-1e3, 1e3], [10**j, 10**j], 'k:', lw=1, zorder=-5)
    for j in range(len(wh)):
        ax.plot([wh[j], wh[j]], [10**-1, 10**9],
                c=cmap1(renormalize(biasav[i][ix[wh]])[j]), lw=0.5, zorder=-1)

# t-tests
stringy = ['(e)', '(f)']
for i in range(2):
    ax = [ax3, ax6][i]
    DF = DFs[i]
    ix = idxs[i]
    bi = bias[i]
    wh = whs[i]
    ax.scatter(range(len(ix)), -np.log(DF.WilcoxP[ix]),
                c='k', s=25)
    ax.scatter(np.arange(len(ix))[wh], -np.log(DF.WilcoxP[ix[wh]]),
                c=renormalize(biasav[i][ix[wh]]), vmin=0, vmax=1,
                cmap=cmap1, edgecolor='k', s=200, lw=2)
    ax.set_ylim([0, 5])
    ax.set_xlim([ax.get_xlim()[0], ax.get_xlim()[1]])
    ax.plot([-1e3, 1e3], [-np.log(0.01), -np.log(0.01)], 'k:', lw=0.5)
    ax.plot([-1e3, 1e3], [-np.log(0.05), -np.log(0.05)], 'k:', lw=0.5)
    ax.plot([-1e3, 1e3], [-np.log(0.1), -np.log(0.1)], 'k:', lw=0.5)
    ax.plot([-1e3, 1e3], [-np.log(0.2), -np.log(0.2)], 'k:', lw=0.5)
    ax.plot([-1e3, 1e3], [-np.log(0.5), -np.log(0.5)], 'k:', lw=0.5)
    if i == 1:
        ax.text(0, -np.log(0.01), 'p-value = 0.01', va='bottom')
        ax.text(0, -np.log(0.05), 'p-value = 0.05', va='bottom')
        ax.text(0, -np.log(0.1), 'p-value = 0.1', va='bottom')
        ax.text(0, -np.log(0.2), 'p-value = 0.2', va='bottom')
        ax.text(0, -np.log(0.5), 'p-value = 0.5', va='bottom')
    ax.text(0.02, 0.98, stringy[i], fontsize=FS, va='top', transform=ax.transAxes)
    for j in range(len(wh)):
        ax.plot([wh[j], wh[j]], [-np.log(np.array(DF.WilcoxP[ix[wh]])[j]), 1e3], c=cmap1(renormalize(biasav[i][ix[wh]])[j]), lw=0.5, zorder=-1)

# Other stuff
stringy = ['Exploration bias', 'Amount of data', r'Wilcoxon'+'\n'+r'$-$log$(p)$']
sty = ['NORMAL PC DATA', 'SHUFFLED PC DATA']
for i in range(2):
    for j in range(3):
        ax = np.array([(ax1, ax4), (ax2, ax5), (ax3, ax6)]).T[i][j]
        DF = DFs[i]
        ax.set_xlim([-2, len(DF)+1])
        ax.tick_params(axis='both', which='major', labelsize=13)
        ax.tick_params(axis='both', which='minor', labelsize=13)
        if j != 2:
            ax.set_xticks([])
        if j == 0:
            ax.set_title(sty[i], fontsize=FS)
        if j == 2:
            ax.set_xlabel('Index cluster'+'\n'+'(sorted by exploration bias)',
                          fontsize=FS)
        if i == 1:
            ax.set_yticks([])
        if i == 0:
            ax.set_ylabel(stringy[j], fontsize=FS)
ax1.set_yticks([0, 0.5, 1, 2, 4, 8])
ax1.set_yticklabels([0, 0.5, 1, 2, 4, 8])

# Determine rank-biserial correlation
import scipy.stats as ss
signornot = np.zeros(len(bias[0][0]))
signornot[whs[0]] = 1
size = np.array(DFs[0].AmountData)
sizerank = ss.rankdata(size)
corr = ss.pointbiserialr(signornot, sizerank)[0]

fig.tight_layout()
plt.savefig(Path_Figsave+'Figure_5.png', bbox_inches='tight', dpi=200)
