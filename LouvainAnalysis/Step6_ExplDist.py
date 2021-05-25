# ----------------------------------------------------------------- #
# About script
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
savepath = '/Users/mmdekker/Documents/Werk/Data/Neuro/Processed/'

# ----------------------------------------------------------------- #
# Calculations
# ----------------------------------------------------------------- #

Expl = np.array(pd.read_pickle(savepath+'Expl.pkl')).T[0]
Clus_s = np.array(pd.read_pickle(savepath+'Clus_s.pkl')).T[0]
Clus_sh = np.array(pd.read_pickle(savepath+'Clus_sh.pkl')).T[0]
DF_meta_s = pd.read_csv(savepath+'ClusterAnalysis_s.csv')

Avperc = len(np.where(Expl >= 2)[0])/len(Expl)
Uni_s = np.unique(Clus_s)
Uni_sh = np.unique(Clus_sh)

#%%
# ----------------------------------------------------------------- #
# Iterative randomization
# ----------------------------------------------------------------- #

Means = []
STDs = []
for mc in tqdm(range(100)):
    expl_new = Expl[np.random.choice(len(Expl), size=len(Expl))]
    biases = []
    for c in Uni_s:
        wh = np.where(Clus_s == c)[0]
        bias = len(np.where(expl_new[wh] >= 2)[0])/(1e-9+len(wh))/Avperc
        biases.append(bias)
    Means.append(np.mean(biases))
    STDs.append(np.std(biases))

pd.DataFrame(Means).to_csv(savepath+'MC_means.csv')
pd.DataFrame(STDs).to_csv(savepath+'MC_stds.csv')
