# ----------------------------------------------------------------- #
# About script
# Figure 4: exploration biases across clusters
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
savepath = '/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/Processed/'

# ----------------------------------------------------------------- #
# Input
# ----------------------------------------------------------------- #

RES = 10
TAU = 30
space = np.linspace(-12, 12, RES)

# ----------------------------------------------------------------- #
# Read data
# ----------------------------------------------------------------- #

Series = pd.read_pickle(savepath+'TotalSeries.pkl')
SE = np.array(Series).T
PC1_hip, PC1_par, PC1_pfc, PC2_hip, PC2_par, PC2_pfc = SE[:6]
PC1s_hip, PC1s_par, PC1s_pfc, PC2s_hip, PC2s_par, PC2s_pfc = SE[6:12]
PC1sh_hip, PC1sh_par, PC1sh_pfc, PC2sh_hip, PC2sh_par, PC2sh_pfc = SE[12:18]
Expl, ROD, cell_s, cluster_s, cell_sh, cluster_sh, cell, cluster = SE[18:]
AvExpl = len(Expl[Expl > 1]) / len(Expl[Expl >= 1])
PC1_hip, PC1_par, PC1_pfc, PC2_hip, PC2_par, PC2_pfc, PC1s_hip, PC1s_par, PC1s_pfc, PC2s_hip, PC2s_par, PC2s_pfc, PC1sh_hip, PC1sh_par, PC1sh_pfc, PC2sh_hip, PC2sh_par, PC2sh_pfc = np.array([PC1_hip, PC1_par, PC1_pfc, PC2_hip, PC2_par, PC2_pfc, PC1s_hip, PC1s_par, PC1s_pfc, PC2s_hip, PC2s_par, PC2s_pfc, PC1sh_hip, PC1sh_par, PC1sh_pfc, PC2sh_hip, PC2sh_par, PC2sh_pfc]).astype(float)

DF_meta = pd.read_csv(savepath+'ClusterAnalysis.csv')
biases = np.array([DF_meta.Explbias_r1,
                   DF_meta.Explbias_r2,
                   DF_meta.Explbias_r3,
                   DF_meta.Explbias_r4,
                   DF_meta.Explbias_r5,
                   DF_meta.Explbias_r6])
avbias = np.array(DF_meta.Explbias)

amounts = []
for j in range(len(DF_meta)):
    amount = np.array([DF_meta.Perc_r1[j],
                       DF_meta.Perc_r2[j],
                       DF_meta.Perc_r3[j],
                       DF_meta.Perc_r4[j],
                       DF_meta.Perc_r5[j],
                       DF_meta.Perc_r6[j]])
    amounts.append(amount)
amounts = np.array(amounts).T

# Filter out near-empty ones:
biases2 = np.copy(biases)
for i in range(len(amounts[0])):
    mean = np.mean(amounts[:, i])
    std = np.std(amounts[:, i])
    for j in range(len(amounts)):
        if amounts[j, i] < mean - std:
            biases2[j, i] = np.nan
idx = np.argsort(avbias)

# With smoothening
DF_meta_s = pd.read_csv(savepath+'ClusterAnalysis_s.csv')
biases_s = np.array([DF_meta_s.Explbias_r1,
                     DF_meta_s.Explbias_r2,
                     DF_meta_s.Explbias_r3,
                     DF_meta_s.Explbias_r4,
                     DF_meta_s.Explbias_r5,
                     DF_meta_s.Explbias_r6])
avbias_s = np.array(DF_meta_s.Explbias)

amounts_s = []
for j in range(len(DF_meta_s)):
    amount = np.array([DF_meta_s.Perc_r1[j],
                       DF_meta_s.Perc_r2[j],
                       DF_meta_s.Perc_r3[j],
                       DF_meta_s.Perc_r4[j],
                       DF_meta_s.Perc_r5[j],
                       DF_meta_s.Perc_r6[j]])
    amounts_s.append(amount)
amounts_s = np.array(amounts_s).T

# Filter out near-empty ones:
biases2_s = np.copy(biases_s)
for i in range(len(amounts_s[0])):
    mean = np.mean(amounts_s[:, i])
    std = np.std(amounts_s[:, i])
    for j in range(len(amounts_s)):
        if amounts_s[j, i] < mean - std:
            biases2_s[j, i] = np.nan
idx_s = np.argsort(avbias_s)

# With shuffling
DF_meta_sh = pd.read_csv(savepath+'ClusterAnalysis_sh.csv')
biases_sh = np.array([DF_meta_sh.Explbias_r1,
                      DF_meta_sh.Explbias_r2,
                      DF_meta_sh.Explbias_r3,
                      DF_meta_sh.Explbias_r4,
                      DF_meta_sh.Explbias_r5,
                      DF_meta_sh.Explbias_r6])
avbias_sh = np.array(DF_meta_sh.Explbias)

amounts_sh = []
for j in range(len(DF_meta_sh)):
    amount = np.array([DF_meta_sh.Perc_r1[j],
                       DF_meta_sh.Perc_r2[j],
                       DF_meta_sh.Perc_r3[j],
                       DF_meta_sh.Perc_r4[j],
                       DF_meta_sh.Perc_r5[j],
                       DF_meta_sh.Perc_r6[j]])
    amounts_sh.append(amount)
amounts_sh = np.array(amounts_sh).T

# Filter out near-empty ones:
biases2_sh = np.copy(biases_sh)
for i in range(len(amounts_sh[0])):
    mean = np.mean(amounts_sh[:, i])
    std = np.std(amounts_sh[:, i])
    for j in range(len(amounts_sh)):
        if amounts_sh[j, i] < mean - std:
            biases2_sh[j, i] = np.nan
idx_sh = np.argsort(avbias_sh)

     #%%   
# ----------------------------------------------------------------- #
# Plot biases
# ----------------------------------------------------------------- #


def renormalize(vec):
    vec = vec - 0.5
    vec[vec < -0.1] = -0.1
    vec[vec > 1.1] = 1.1
    return vec


cmap1 = plt.cm.RdBu_r
cols = ['steelblue', 'tomato', 'forestgreen']
fig, ((ax1, ax3, ax5), (ax2, ax4, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(25, 20))

# Column 1: no smoothening
ax1.text(0.02, 0.98, '(a) Exploration bias (sorted)', fontsize=14,
         transform=ax1.transAxes, va='top')
ax2.text(0.02, 0.98, '(d) Amount of data points inside cluster', fontsize=14,
         transform=ax2.transAxes, va='top')

ax1.plot(np.arange(len(DF_meta)), DF_meta.Explbias[idx], c='k',
         label='Exploring', zorder=-1)
ax1.scatter(np.arange(len(DF_meta)), DF_meta.Explbias[idx],
            c=renormalize(avbias[idx]),
            cmap=cmap1, edgecolor='k', lw=1, s=75)

#for i in range(len(biases)):
#    ax1.scatter(range(len(DF_meta)), biases[i][idx],
#                c='silver', s=5, zorder=-1, marker='s')#*amounts[:, i], zorder=-1)
for i in range(len(biases)):
    ax1.scatter(range(len(DF_meta)), biases[i][idx],
                c=renormalize(biases[i][idx]), cmap=cmap1, s=30, zorder=-1, marker='s')#*amounts[:, i], zorder=-1)


ax1.plot([-1e3, 1e3], [1, 1], 'k:', lw=1, zorder=-1)
ax1.set_xlim([-1, len(DF_meta)+1])
ax1.set_ylim([0, 4])
ax2.set_xlabel('Index cluster (sorted by exploration bias)')
#ax.set_ylabel('Bias towards exploring')
ax2.scatter(range(len(DF_meta)), DF_meta.AmountData[idx],
            c=renormalize(avbias[idx]),
            cmap=cmap1, edgecolor='k', s=100)
ax2.semilogy(range(len(DF_meta)), DF_meta.AmountData[idx], 'k', zorder=-1)
#ax2.set_ylabel('Amount of data in cluster')
ax2.set_ylim([ax2.get_ylim()[0], ax2.get_ylim()[1]])
ax2.set_xlim([ax2.get_xlim()[0], ax2.get_xlim()[1]])
for i in range(10):
    ax2.plot([-1e3, 1e3], [10**i, 10**i], 'k:', lw=0.5, zorder=-1)

# Column 2: with smoothening
ax3.text(0.02, 0.98, '(b) Exploration bias (sorted) - with smoothening', fontsize=14,
         transform=ax3.transAxes, va='top')
ax4.text(0.02, 0.98, '(e) Amount of data points inside cluster - with smoothening', fontsize=14,
         transform=ax4.transAxes, va='top')

ax3.plot(np.arange(len(DF_meta_s)), DF_meta_s.Explbias[idx_s], c='k',
         label='Exploring', zorder=-1)
ax3.scatter(np.arange(len(DF_meta_s)), DF_meta_s.Explbias[idx_s],
            c=renormalize(avbias_s[idx_s]),
            cmap=cmap1, edgecolor='k', lw=1, s=75)

# for i in range(len(biases_s)):
#     ax3.scatter(range(len(DF_meta_s)), biases_s[i][idx_s],
#                 c='silver', s=5, zorder=-1, marker='s')#*amounts[:, i], zorder=-1)
for i in range(len(biases_s)):
    ax3.scatter(range(len(DF_meta_s)), biases_s[i][idx_s],
                c=renormalize(biases_s[i][idx_s]), cmap=cmap1, s=30, zorder=-1, marker='s')#*amounts[:, i], zorder=-1)

ax3.plot([-1e3, 1e3], [1, 1], 'k:', lw=1, zorder=-1)
ax3.set_xlim([-1, len(DF_meta_s)+1])
ax3.set_ylim([0, 4])
ax4.set_xlabel('Index cluster (sorted by exploration bias)')
#ax.set_ylabel('Bias towards exploring')
ax4.scatter(range(len(DF_meta_s)), DF_meta_s.AmountData[idx_s],
            c=renormalize(avbias_s[idx_s]),
            cmap=cmap1, edgecolor='k', s=100)
ax4.semilogy(range(len(DF_meta_s)), DF_meta_s.AmountData[idx_s], 'k', zorder=-1)
#ax2.set_ylabel('Amount of data in cluster')
ax4.set_ylim([ax4.get_ylim()[0], ax4.get_ylim()[1]])
ax4.set_xlim([ax4.get_xlim()[0], ax4.get_xlim()[1]])
for i in range(10):
    ax4.plot([-1e3, 1e3], [10**i, 10**i], 'k:', lw=0.5, zorder=-1)

# Column 3: shuffled
ax5.text(0.02, 0.98, '(c) Exploration bias (sorted) - with smoothening/shuffling', fontsize=14,
         transform=ax5.transAxes, va='top')
ax6.text(0.02, 0.98, '(f) Amount of data points inside cluster - with smoothening/shuffling', fontsize=14,
         transform=ax6.transAxes, va='top')

ax5.plot(np.arange(len(DF_meta_sh)), DF_meta_sh.Explbias[idx_sh], c='k',
         label='Exploring', zorder=-1)
ax5.scatter(np.arange(len(DF_meta_sh)), DF_meta_sh.Explbias[idx_sh],
            c=renormalize(avbias_sh[idx_sh]),
            cmap=cmap1, edgecolor='k', lw=1, s=75)

for i in range(len(biases_sh)):
    ax5.scatter(range(len(DF_meta_sh)), biases_sh[i][idx_sh],
                c='silver', s=5, zorder=-1, marker='s')#*amounts[:, i], zorder=-1)
for i in range(len(biases_sh)):
    ax5.scatter(range(len(DF_meta_sh)), biases2_sh[i][idx_sh],
                c=renormalize(biases2_sh[i][idx_sh]), cmap=cmap1, s=30, zorder=-1, marker='s')#*amounts[:, i], zorder=-1)


ax5.plot([-1e3, 1e3], [1, 1], 'k:', lw=1, zorder=-1)
ax5.set_xlim([-1, len(DF_meta_sh)+1])
ax5.set_ylim([0, 4])
ax6.set_xlabel('Index cluster (sorted by exploration bias)')
#ax.set_ylabel('Bias towards exploring')
ax6.scatter(range(len(DF_meta_sh)), DF_meta_sh.AmountData[idx_sh],
            c=renormalize(avbias_sh[idx_sh]),
            cmap=cmap1, edgecolor='k', s=100)
ax6.semilogy(range(len(DF_meta_sh)), DF_meta_sh.AmountData[idx_sh], 'k', zorder=-1)
#ax2.set_ylabel('Amount of data in cluster')
ax6.set_ylim([ax6.get_ylim()[0], ax6.get_ylim()[1]])
ax6.set_xlim([ax6.get_xlim()[0], ax6.get_xlim()[1]])
for i in range(10):
    ax6.plot([-1e3, 1e3], [10**i, 10**i], 'k:', lw=0.5, zorder=-1)

ax7.plot(np.array(DF_meta.WilcoxP)[idx], 'ko')
ax8.plot(np.array(DF_meta_s.WilcoxP)[idx_s], 'ko')
ax9.plot(np.array(DF_meta_sh.WilcoxP)[idx_sh], 'ko')
axes2 = [ax7, ax8, ax9]
stringy = ['(g) P values Wilcoxon t-test',
           '(h) P values Wilcoxon t-test (smooth)',
           '(i) P values Wilcoxon t-test (smooth+shuffle)']
for i in range(3):
    ax = axes2[i]
    ax.set_ylim([0, 1])
    ax.set_xlim([ax.get_xlim()[0], ax.get_xlim()[1]])
    ax.plot([-1e3, 1e3], [0.05, 0.05], 'k:', lw=0.5)
    ax.plot([-1e3, 1e3], [0.1, 0.1], 'k:', lw=0.5)
    ax.plot([-1e3, 1e3], [0.25, 0.25], 'k:', lw=0.5)
    ax.text(0, 0.05, 'p-value = 0.05', va='center')
    ax.text(0, 0.1, 'p-value = 0.1', va='center')
    ax.text(0, 0.25, 'p-value = 0.25', va='center')
    ax.text(0.02, 0.98, stringy[i], fontsize=14, va='top', transform=ax.transAxes)

fig.tight_layout()
#plt.savefig(Path_Figsave+'Figure_4.png', bbox_inches='tight', dpi=200)

#%%
