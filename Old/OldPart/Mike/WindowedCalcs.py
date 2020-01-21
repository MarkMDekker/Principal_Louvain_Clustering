# --------------------------------------------------------------------- #
# Preambule
# --------------------------------------------------------------------- #

import numpy as np
from scipy.fftpack import rfftfreq, rfft
from tqdm import tqdm
import sys
import scipy.io
import pandas as pd
import warnings
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
exec(open('/Users/mmdekker/Documents/Werk/Scripts/__Other/BrainProject/Mike/Functions.py').read())

# --------------------------------------------------------------------- #
# Calculations
# --------------------------------------------------------------------- #

Series = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Scripts/__Other/BrainProject/Mike/ProcData/Series.pkl'))
#Synchronization, RelativePower, Steps = MeanCorrelation(Series,500,1000)
R, D, C, Cd, PowerR, Steps, Cmats = MeanRmetric(Series,500,2000)

#%% Save stuff

pd.DataFrame(R).to_pickle('/Users/mmdekker/Documents/Werk/Scripts/__Other/BrainProject/Mike/ProcData/R.pkl')
pd.DataFrame(D).to_pickle('/Users/mmdekker/Documents/Werk/Scripts/__Other/BrainProject/Mike/ProcData/D.pkl')
pd.DataFrame(C).to_pickle('/Users/mmdekker/Documents/Werk/Scripts/__Other/BrainProject/Mike/ProcData/C.pkl')
pd.DataFrame(Cd).to_pickle('/Users/mmdekker/Documents/Werk/Scripts/__Other/BrainProject/Mike/ProcData/Cd.pkl')
pd.DataFrame(PowerR).to_pickle('/Users/mmdekker/Documents/Werk/Scripts/__Other/BrainProject/Mike/ProcData/PowerR.pkl')
pd.DataFrame(Steps).to_pickle('/Users/mmdekker/Documents/Werk/Scripts/__Other/BrainProject/Mike/ProcData/Steps.pkl')
np.save('/Users/mmdekker/Documents/Werk/Scripts/__Other/BrainProject/Mike/ProcData/Cmats',Cmats)

#%%
# --------------------------------------------------------------------- #
# Process frequency bands - Version Deb
# --------------------------------------------------------------------- #

Power_proc = []
for i in range(len(PowerR[0])):
    Sery = np.copy(PowerR[:,i])
    Sery = Sery - np.nanmean(Sery)
    Sery = Sery / np.nanstd(Sery)
    Power_proc.append(Sery)
Power_proc = np.array(Power_proc)

# --------------------------------------------------------------------- #
# Plot time series
# --------------------------------------------------------------------- #
WM =25
fig,(ax1,ax2) = plt.subplots(2,1,figsize=(12,12),sharex=True)
#ax1.plot(Steps/1000.,windowmean(R,WM),'k',lw=1,label='Average R matrix')
#ax1.plot(Steps/1000.,windowmean(D,WM),'tomato',lw=1,label='Average of deviations from mean in R matrix')
ax1.plot(Steps/1000.,windowmean(C,WM),'blue',lw=1,label='Correlation (Pearson)')
ax1.plot(Steps/1000.,windowmean(Cd,WM),'forestgreen',lw=1,label='Average of deviations from mean in correlation (Pearson)')
ax1.set_ylabel('Metrics')
ax1.legend()
c=ax2.pcolormesh(Steps/1000.,range(100),Power_proc,vmin=-1,vmax=1,
               cmap = plt.cm.coolwarm)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Frequency (Hz)')
plt.colorbar(c,orientation='horizontal',label='Normalized relative power')
fig.tight_layout

##%%
## --------------------------------------------------------------------- #
## Process frequency bands
## --------------------------------------------------------------------- #
#
#Power_proc = []
#for i in range(len(RelativePower[0])):
#    Sery = np.copy(RelativePower[:,i])
#    Sery = Sery - np.nanmean(Sery)
#    Sery = Sery / np.nanstd(Sery)
#    Power_proc.append(Sery)
#Power_proc = np.array(Power_proc)
#
## --------------------------------------------------------------------- #
## Plot time series
## --------------------------------------------------------------------- #
#
#fig,(ax1,ax2) = plt.subplots(2,1,figsize=(12,12),sharex=True)
#ax1.plot(Steps/1000.,Synchronization,'k',lw=2)
#ax1.set_ylabel('Average of correlation matrix')
#c=ax2.pcolormesh(Steps/1000.,range(100),Power_proc,vmin=-1,vmax=1,
#               cmap = plt.cm.coolwarm)
#ax2.set_xlabel('Time (s)')
#ax2.set_ylabel('Frequency (Hz)')
#plt.colorbar(c,orientation='horizontal',label='Normalized relative power')
#fig.tight_layout