# ----------------------------------------------------------------- #
# About script
# Figure 4: Correlation matrix + scatters + histograms
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import pandas as pd
import scipy.stats
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pyplot as plt
path = ('/Users/mmdekker/Documents/Werk/Data/Neuro/Processed/')
Path_Figsave = ('/Users/mmdekker/Documents/Werk/Figures/Papers/2020_Brain/')

# ----------------------------------------------------------------- #
# Read data
# ----------------------------------------------------------------- #

DF_meta = pd.read_csv(path+'ClusterAnalysis_s.csv', index_col=0)
DF_meta['RodentSpecific'] = np.array(np.sum(np.abs(1/11- DF_meta[['Perc_r1', 'Perc_r2', 'Perc_r3',
                                                                 'Perc_r4', 'Perc_r5', 'Perc_r6',
                                                                 'Perc_r7', 'Perc_r8', 'Perc_r9',
                                                                 'Perc_r10', 'Perc_r11']]), axis=1))/(20/11)
Relevant = ['AmountData', 'Explbias', 'Absbias', 'AvResTime', 'Centr_to_origin']
DF_meta2 = DF_meta[Relevant]
spear = DF_meta2.corr(method='spearman')
spear.to_csv(path+'Spearman.csv')
keys = spear.keys()
keys2 = ['Amount of'+'\n'+'data', 'Exploration'+'\n'+'bias', 'Abs.'+'\n'+'Exploration'+'\n'+'bias', 'Average'+'\n'+'residence'+'\n'+'time', 'Distance to'+'\n'+'origin']
spear = np.array(spear)
N = len(spear)

TS = []
for i in range(len(Relevant)):
    TS.append(np.array(DF_meta2[[Relevant[i]]]))

pvals = np.zeros(shape=spear.shape)
for i in range(len(Relevant)):
    for j in range(len(Relevant)):
        pvals[i, j] = scipy.stats.spearmanr(TS[i], TS[j])[1]

# ----------------------------------------------------------------- #
# Plot
# ----------------------------------------------------------------- #

fig, axes = plt.subplots(figsize=(15, 5))
gs = gridspec.GridSpec(3, 3)
gs.update(wspace=0.2, hspace=0.2)
ax1 = plt.subplot(gs[0:3, 1])
ax2 = plt.subplot(gs[0:3, 0])
ax3 = plt.subplot(gs[0, 2])
ax4 = plt.subplot(gs[1, 2])
ax5 = plt.subplot(gs[2, 2])

cmap = plt.cm.RdBu_r
ax1.scatter(DF_meta.Absbias,DF_meta.Centr_to_origin, s=np.array(np.log(DF_meta.AmountData))**4/200,
            edgecolor='k', lw=0.5, c='forestgreen')
# ax1.annotate(str(DF_meta.AmountData[2])+' datapoints',
#              xy=(DF_meta.Absbias[2], DF_meta.Centr_to_origin[2]),  # theta, radius
#              xytext=(0.5, 0.3),
#              textcoords='figure fraction',
#              arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
#              horizontalalignment='left',
#              verticalalignment='bottom',
#              )
# ax1.annotate(str(DF_meta.AmountData[47])+' datapoints',
#              xy=(DF_meta.Absbias[37], DF_meta.Centr_to_origin[37]),  # theta, radius
#              xytext=(0.6, 0.5),
#              textcoords='figure fraction',
#              arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
#              horizontalalignment='right',
#              verticalalignment='bottom',
#              )
ax1.set_xlabel('Absolute exploration bias')
ax1.set_ylabel('Cluster distance to origin (a.u.)')
ax2.pcolormesh(spear, vmin=-1, vmax=1, cmap=cmap)
for i in range(len(keys)):
    for j in range(len(keys)):
        if i != j:
            if np.abs(pvals[i, j]) < 0.05:
                ax2.text(i+0.5, j+0.5, np.round(spear[i, j], 2), c='k', ha='center', va='center', fontsize=10)
            else:
                ax2.text(i+0.5, j+0.5, np.round(spear[i, j], 2), c='grey', ha='center', va='center', fontsize=10)
ax2.scatter([2.5], [4.5], s=700, facecolors='none', edgecolor='k', lw=2)
ax2.scatter([4.5], [2.5], s=700, facecolors='none', edgecolor='k', lw=2)
ax2.set_ylim([0, len(keys)])
ax2.set_yticks(np.arange(N)+0.5)
ax2.set_yticklabels(keys2)
ax2.set_xlim([0, len(keys)])
ax2.set_xticks(np.arange(N)+0.5)
ax2.set_xticklabels(keys2, rotation=45, ha='right', va='top')
#ax2.set_title('(a)', fontsize=13)
#ax1.set_title('(b)', fontsize=13)
ax1.text(-0.02, 1.06, '(b) Relation $|\zeta_b|$ and distance to origin', fontsize=13, ha='left', va='top',
         transform=ax1.transAxes)
ax2.text(-0.02, 1.06, '(a) Correlations among cluster metrics', fontsize=13, ha='left', va='top',
         transform=ax2.transAxes)
#ax1.set_yticks([])

N = 25
ax3.hist(DF_meta['AvResTime'], bins=N, alpha=0.75)
ax4.hist(DF_meta['InterReg'], bins=N, alpha=0.75)
ax5.hist(DF_meta['RodentSpecific'], bins=N, alpha=0.75)

avs = [np.mean(DF_meta['AvResTime']),
       np.mean(DF_meta['InterReg']),
       np.mean(DF_meta['RodentSpecific'])]

string = ['(c) Average residence time',
          '(d) Interregionality',
          '(e) Mouse specificity']
for i in range(3):
    ax = [ax3, ax4, ax5][i]
    ax.set_yticks([])
    ax.set_ylim([ax.get_ylim()[0], ax.get_ylim()[1]+5])
    ax.text(0.02, 0.98, string[i], fontsize=13, ha='left', va='top',
            transform=ax.transAxes)
    ax.plot([avs[i], avs[i]], [-1e3, 1e3], 'silver')
    ax.text(avs[i], 6, 'Mean: '+str(np.round(avs[i], 2)))
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.tick_params(axis='both', which='minor', labelsize=9)
    ax.set_ylabel('Density')

fig.tight_layout()
plt.savefig(Path_Figsave+'Figure_4.png', bbox_inches='tight', dpi=200)
