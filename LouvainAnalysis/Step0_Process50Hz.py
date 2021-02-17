# ----------------------------------------------------------------- #
# About script
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
path = ('/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/'
        'Output/ResidualSeries')

# ----------------------------------------------------------------- #
# Read data
# ----------------------------------------------------------------- #

REG = 'PAR'
SUB = '346111'
Series = np.array(pd.read_pickle(path+'/F_'+REG+'_'+SUB+'.pkl'))

# ----------------------------------------------------------------- #
# Process
# ----------------------------------------------------------------- #

Series2 = np.copy(Series)
Series2[24] = np.mean(np.array([Series[23], Series[25]]), axis=0)
covmat = Series2.dot(Series2.T)
vals, vecs = np.linalg.eig(covmat)
plt.plot(vecs[:, 0])

#%%
# ----------------------------------------------------------------- #
# Save
# ----------------------------------------------------------------- #

pd.DataFrame(Series2).to_pickle(path+'/F_'+REG+'_'+SUB+'.pkl')
