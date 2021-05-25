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

# ----------------------------------------------------------------- #
# Calculations
# ----------------------------------------------------------------- #

pathname = '/Users/mmdekker/Documents/Werk/Data/Neuro/Output/'
SUBs = ['279418', '279419', '339295', '339296', '340233', '341319',
        '341321', '346110', '346111', '347640', '347641']
REGs = ['HIP', 'PAR', 'PFC']
TYPEs = ['', 'T']
Am = []
Bm = []
As = []
Bs = []
Nams = []
for sub in tqdm(SUBs):
    for reg in REGs:
        for typ in TYPEs:
            Name = reg+'_'+sub+typ
            Nams.append(Name)
            Aar = np.array(pd.read_csv(pathname+'A/'+Name+'.csv', index_col=0))
            Bar = np.array(pd.read_csv(pathname+'B/'+Name+'.csv', index_col=0))
            Am.append(np.nanmean(Aar))
            Bm.append(np.nanmean(Bar))
            As.append(np.nanstd(Aar))
            Bs.append(np.nanstd(Bar))

# ----------------------------------------------------------------- #
# To DF
# ----------------------------------------------------------------- #

DF = {}
DF['Dataset'] = Nams
DF['A_mean'] = Am
DF['A_std'] = As
DF['B_mean'] = Bm
DF['B_std'] = Bs
pd.DataFrame(DF).to_csv(pathname+'Fit.csv')