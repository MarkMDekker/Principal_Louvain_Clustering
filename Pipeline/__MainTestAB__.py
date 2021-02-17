# ----------------------------------------------------------------- #
# About script
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import numpy as np
import matplotlib.pyplot as plt
from Class_TestAB import Tester
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------- #
# Run Class
# ----------------------------------------------------------------- #

# for sub in ['279418', '279419', '339295', '339296', '340233', '341319',
#             '341321', '346110', '346111', '347640', '347641']:
for sub in ['279418']:
    for reg in ['HIP']:
        for st in ['TRAINING']:
            params_input = {'SUBPART': 'NO',
                            'REGION': reg,
                            'TYPE': st,
                            'SUBJECT': sub}
            Class = Tester(params_input)
            Class.preprocess()
            Class.determine_residual_series()
            Class.save_to_file()

path = '/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/Output/ResidualSeries/HIP_279418.pkl'
resid = np.array(pd.read_pickle(path))

#%%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
ax1.hist(Class.A.flatten(), bins=25)
ax2.hist(Class.B.flatten(), bins=25)
ax1.set_xlabel('a (= exponent)')
ax2.set_xlabel('b (= multiplicative factor)')
