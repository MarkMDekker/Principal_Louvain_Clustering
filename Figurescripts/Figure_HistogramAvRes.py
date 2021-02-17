# ----------------------------------------------------------------- #
# About script
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------- #
# Read data
# ----------------------------------------------------------------- #

DF = pd.read_csv('/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/Processed/ClusterAnalysis.csv')
DF_s = pd.read_csv('/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/Processed/ClusterAnalysis_s.csv')
DF_sh = pd.read_csv('/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/Processed/ClusterAnalysis_sh.csv')

# ----------------------------------------------------------------- #
# Plot histograms
# ----------------------------------------------------------------- #

fig, ax = plt.subplots(figsize=(12, 5))
ax.hist(DF.AvResTime, bins=np.arange(0, 600, 10), density=True, label='Raw', alpha=0.5)
ax.hist(DF_s.AvResTime, bins=np.arange(0, 600, 10), density=True, label='Smooth', alpha=0.5)
ax.hist(DF_sh.AvResTime, bins=np.arange(0, 600, 10), density=True, label='Smooth+Shuffle', alpha=0.5)
ax.legend(loc='best')
ax.set_xlabel('Average residence time (ms)')
ax.set_ylabel('Density')