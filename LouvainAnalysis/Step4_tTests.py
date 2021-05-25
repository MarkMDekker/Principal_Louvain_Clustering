# ----------------------------------------------------------------- #
# About script
# Performs Wilcoxon t test
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import pandas as pd
from scipy.stats import wilcoxon
import numpy as np
import matplotlib.pyplot as plt
savepath = '/Users/mmdekker/Documents/Werk/Data/Neuro/Processed/'

# ----------------------------------------------------------------- #
# Read data
# ----------------------------------------------------------------- #

SUBs = ['279418', '279419', '339295', '339296', '340233', '341319',
        '341321', '346110', '346111', '347640', '347641']
for file in ['ClusterAnalysis.csv', 'ClusterAnalysis_s.csv', 'ClusterAnalysis_sh.csv']:
    DF_meta = pd.read_csv(savepath+file, index_col=0)
    biases_s = np.array([DF_meta.Explbias_r1,
                         DF_meta.Explbias_r2,
                         DF_meta.Explbias_r3,
                         DF_meta.Explbias_r4,
                         DF_meta.Explbias_r5,
                         DF_meta.Explbias_r6,
                         DF_meta.Explbias_r7,
                         DF_meta.Explbias_r8,
                         DF_meta.Explbias_r9,
                         DF_meta.Explbias_r10,
                         DF_meta.Explbias_r11])
    avbias_s = np.array(DF_meta.Explbias)
    N = len(DF_meta)

    # ----------------------------------------------------------------- #
    # Read data
    # ----------------------------------------------------------------- #

    # ps = np.zeros(shape=(N, N))+np.nan
    # for i in range(N):
    #     for j in range(N):
    #         if i != j:
    #             try:
    #                 w, p  = wilcoxon(biases_s[:, i], biases_s[:, j])
    #                 ps[i, j] = p
    #             except:
    #                 continue
    PERCS = DF_meta[['Perc_r1', 'Perc_r2', 'Perc_r3', 'Perc_r4', 'Perc_r5',
                     'Perc_r6', 'Perc_r7', 'Perc_r8', 'Perc_r9', 'Perc_r10',
                     'Perc_r11']]
    PERCS = np.array(PERCS)
    ps = np.zeros(shape=(N))+np.nan
    for i in range(N):
        wh = np.where(PERCS[i] > 0.02)[0] # at least 1 sec
        try:
            w, p  = wilcoxon(biases_s[wh, i]-np.ones(len(wh)))
            ps[i] = p
        except:
            continue
    
    DF_meta['WilcoxP'] = ps
    pd.DataFrame(DF_meta).to_csv(savepath+file)
