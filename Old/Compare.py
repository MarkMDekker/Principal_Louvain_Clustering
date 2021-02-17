# ----------------------------------------------------------------- #
# About script
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.stats as stats

# ----------------------------------------------------------------- #
# Calculations
# ----------------------------------------------------------------- #

old = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/Output/HIP_279418.pkl'))
new = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/Output/ResidualSeries/HIP_279418.pkl'))

#%%
As = []
Bs = []
Names = []
for sub in ['279419']:
    for reg in tqdm(['HIP', 'PFC', 'PAR']):
        for st in ['TRAINING', 'TEST']:
            if st == 'TRAINING':
                Name = reg+'_'+sub
            else:
                Name = reg+'_'+sub+'T'
            A = np.array(pd.read_csv('/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/Output/A/'+Name+'.csv', index_col=0)).flatten()
            B = np.array(pd.read_csv('/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/Output/B/'+Name+'.csv', index_col=0)).flatten()
            As.append(A)
            Bs.append(B)
            Names.append(Name)

#%%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
for i in range(len(As)):
    col = plt.cm.coolwarm(i/len(As))
    
    y = As[i][:10000]
    y = y[~np.isnan(y)]
    density = stats.gaussian_kde(y)
    n, x, _ = ax1.hist(y, bins=25, label=Names[i], alpha=0.25, density=True, color=col)
    ax1.plot(x, density(x), c=col, lw=2, zorder=1e9)
    
    y = Bs[i][:10000]
    y = y[~np.isnan(y)]
    density = stats.gaussian_kde(y)
    n, x, _ = ax2.hist(y, bins=25, label=Names[i], alpha=0.25, density=True, color=col)
    ax2.plot(x, density(x), c=col, lw=2, zorder=1e9)
ax1.legend(loc='best')
ax1.set_xlabel('a (= exponent)')
ax1.set_ylabel('Density')
ax2.set_xlabel('b (= mult factor)')
plt.savefig('../Test.png')