# --------------------------------------------------------------------- #
# Principal Component Analysis
# --------------------------------------------------------------------- #

import pandas as pd
import numpy as np
path_pdata = ('/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/'
              'ProcessedData/')

# --------------------------------------------------------------------- #
# Read data
# --------------------------------------------------------------------- #

p = 'HIP'
ex = '600000_0'
s = p+'/'+ex
ProcessedPower = np.array(pd.read_pickle(path_pdata+'ProcessedPower/'+s+'.pkl'))

# --------------------------------------------------------------------- #
# Full PCA calculations
# --------------------------------------------------------------------- #

vals, vecs = np.linalg.eig(ProcessedPower.dot(ProcessedPower.T))
idx = vals.argsort()[::-1]
vals = vals[idx]
vecs = vecs[:, idx]
vals_num = np.copy(vals/np.sum(vals))
EOFs = []
PCs = []
for i in range(len(ProcessedPower)):
    EOFs.append(vecs[:, i])
    PCs.append(vecs[:, i].dot(ProcessedPower))

# --------------------------------------------------------------------- #
# Savings and rescaling
# --------------------------------------------------------------------- #

pd.DataFrame(PCs).to_pickle(path_pdata+'PCs/'+s+'.pkl')
pd.DataFrame(EOFs).to_pickle(path_pdata+'EOFs/'+s+'.pkl')
pd.DataFrame(vals_num).to_pickle(path_pdata+'EVs/'+s+'.pkl')
