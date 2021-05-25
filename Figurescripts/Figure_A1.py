# -------------------------------------------------------------------------- #
# Preambule
# -------------------------------------------------------------------------- #

import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt

Path_Figsave = ('/Users/mmdekker/Documents/Werk/Figures/Papers/2020_Brain/')

# -------------------------------------------------------------------------- #
# Read data
# -------------------------------------------------------------------------- #

RawData = scipy.io.loadmat('/Users/mmdekker/Documents/Werk/Data/Neuro/newdata/279418_Training.mat')
DataEEG = RawData['EEG'][0][0][9]
EOF1hip = np.array(pd.read_csv('/Users/mmdekker/Documents/Werk/Data/Neuro/Processed/EOF1/HIP.csv', index_col=0).Average)
ResidS = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Neuro/Output/ResidualSeries/HIP_279418.pkl'))
PC = EOF1hip.dot(ResidS)

T0 = 457023
T1 = T0+1500
T = np.arange(T0, T1)

# -------------------------------------------------------------------------- #
# Plot
# -------------------------------------------------------------------------- #

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax1.plot(T, DataEEG[0][T0:T1])
ax2.plot(T, PC[T0:T1])

ax1.set_title('Example electrode signal and PC time series (mouse 279418, training)', fontsize=17)
ax1.text(0.01, 0.98, '(a) Signal of electrode #1', fontsize=17, transform=ax1.transAxes, va='top', ha='left')
ax2.text(0.01, 0.98, '(b) PC1, hippocampus', fontsize=17, transform=ax2.transAxes, va='top', ha='left')

ax1.set_ylabel('Electrode signal', fontsize=13)
ax2.set_ylabel('PC amplitude', fontsize=13)
ax2.set_xlabel('Time in experiment (ms)', fontsize=13)
fig.tight_layout()
plt.savefig(Path_Figsave+'Figure_A1.png', bbox_inches='tight', dpi=200)
