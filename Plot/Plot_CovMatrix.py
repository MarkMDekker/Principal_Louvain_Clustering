# --------------------------------------------------------------------------- #
# Script to create correlation matrices of residues
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------- #
# Preambules and paths
# --------------------------------------------------------------------- #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import matplotlib as mpl

strings = ['Exploring new','Exploring old','Not exploring']
figdir = '/Users/mmdekker/Documents/Werk/Figures/Neurostuff/Dataset_new/'
datadir = '/Users/mmdekker/Documents/Werk/Data/Braindata/ObjectTest.mat'

# --------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------- #

save = 1

# --------------------------------------------------------------------- #
# Read data
# --------------------------------------------------------------------- #

CorrMatA = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/CorrMat/CorrMatA.pkl'))
CorrMatB = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/CorrMat/CorrMatB.pkl'))
CorrMatC = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/CorrMat/CorrMatC.pkl'))

# --------------------------------------------------------------------- #
# Relative power and synchronization
# --------------------------------------------------------------------- #

fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(12,4.5),sharey=True)
cms = [CorrMatA,CorrMatB,CorrMatC]
mn,mx = (np.nanmin(cms),np.nanmax(cms))
axs = [ax1,ax2,ax3]
for j in range(3):
    ax = axs[j]
    ax.pcolormesh(cms[j],vmin=mn,vmax=mx, cmap=plt.cm.rainbow)
    ax.set_title(strings[j])
    ax.set_xlabel('Channel')
    ax.set_xlim([ax.get_xlim()[0],ax.get_xlim()[1]])
    ax.set_ylim([ax.get_ylim()[0],ax.get_ylim()[1]])
    ax.plot([16,16],[-1e3,1e3],'k',lw=2,zorder=1e9)
    ax.plot([-1e3,1e3],[16,16],'k',lw=2,zorder=1e9)
    ax.plot([24,24],[-1e3,1e3],'k',lw=2,zorder=1e9)
    ax.plot([-1e3,1e3],[24,24],'k',lw=2,zorder=1e9)
ax1.set_ylabel('Channel')
axc     = fig.add_axes([0.24, -0.035, 0.6, 0.03])
norm    = mpl.colors.Normalize(vmin=mn,vmax=mx)
cb1     = mpl.colorbar.ColorbarBase(axc, cmap=plt.cm.rainbow,norm=norm,extend='both',orientation='horizontal')
cb1.set_label('Correlation of fourier residues',fontsize=15)
axc.tick_params(labelsize=13)
fig.tight_layout()
if save==1: plt.savefig('/Users/mmdekker/Documents/Werk/Figures/Neurostuff/Pics_04112019/CorrResidu.png',bbox_inches = 'tight',dpi=200)
plt.show()