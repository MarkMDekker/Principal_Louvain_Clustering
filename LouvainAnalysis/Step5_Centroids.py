# ----------------------------------------------------------------- #
# About script
# Determines centroid differences to origin and appends to DF
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
# Read data
# ----------------------------------------------------------------- #

for dataset in tqdm(['', '_s', '_sh']):
    file = 'ClusterAnalysis'+dataset+'.csv'
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
    
    Series = pd.read_pickle(savepath+'TotalSeries.pkl')
    Clus = np.array(Series['Cluster'+dataset])
    PCs = np.array([Series['PC1'+dataset[1:]+'_hip'],
                    Series['PC2'+dataset[1:]+'_hip'],
                    Series['PC1'+dataset[1:]+'_par'],
                    Series['PC2'+dataset[1:]+'_par'],
                    Series['PC1'+dataset[1:]+'_pfc'],
                    Series['PC2'+dataset[1:]+'_pfc']])
    
    # ----------------------------------------------------------------- #
    # Read data
    # ----------------------------------------------------------------- #
    
    ps = np.zeros(shape=(N))+np.nan
    cen = []
    for i in range(N):
        wh = np.where(Clus == i)[0]
        centr_to_origin = np.sqrt(np.nansum(np.nansum(PCs[:, wh]**2, axis=0))/len(wh))
        cen.append(centr_to_origin)
    
    DF_meta['Centr_to_origin'] = cen
    pd.DataFrame(DF_meta).to_csv(savepath+file)
