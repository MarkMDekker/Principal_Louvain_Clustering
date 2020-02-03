# --------------------------------------------------------------------- #
# Scramble the phases of the data
# --------------------------------------------------------------------- #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from tqdm import tqdm

# --------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------- #

PATH_RAWDATA = '/Users/mmdekker/Documents/Werk/Data/Braindata/ObjectTest.mat'
PATH_PROCDATA = '/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/'

# --------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------- #

ex = 'HR_HIP'

# --------------------------------------------------------------------- #
# Read data
# --------------------------------------------------------------------- #

Power = np.array(pd.read_pickle(PATH_PROCDATA+'ProcessedPower/'+ex+'.pkl'))
Steps = np.array(pd.read_pickle(PATH_PROCDATA+'Steps/'+ex+'.pkl')).T[0]
Synch = np.array(pd.read_pickle(PATH_PROCDATA+'Synchronization/'+ex+'.pkl')
                 ).T[0]
PCs = np.array(pd.read_pickle(PATH_PROCDATA+'PCs/'+ex+'.pkl'))
EOFs = np.array(pd.read_pickle(PATH_PROCDATA+'EOFs/'+ex+'.pkl'))

# --------------------------------------------------------------------- #
# Select data
# --------------------------------------------------------------------- #

mat = scipy.io.loadmat(PATH_RAWDATA)
DataAll = mat['EEG'][0][0]
Time = DataAll[14]
Data_dat = DataAll[15]
Data_re1 = DataAll[17]
Data_re2 = DataAll[18]
Data_re3 = DataAll[19]
Data_re4 = DataAll[25]
Data_re5 = DataAll[30]
Data_re6 = DataAll[39]
Data_vid = DataAll[40].T
ExploringNew = np.where(Data_vid[2] == 1)[0]
ExploringOld = np.where(Data_vid[3] == 1)[0]
NotExploring = np.array(list(np.where(Data_vid[2] != 1)[0]) +
                        list(np.where(Data_vid[3] != 1)[0]))

# --------------------------------------------------------------------- #
# Scramble phases
# --------------------------------------------------------------------- #

inds = np.arange(0, len(Power[0]))
Power_f = np.copy(Power)
for i in range(len(Power_f)):
    ind = np.random.choice(inds)
    inds_new = list(inds[ind:])+list(inds[:ind])
    Power_f[i] = Power_f[i, inds_new]

# --------------------------------------------------------------------- #
# Full PCA calculations
# --------------------------------------------------------------------- #

vals, vecs = np.linalg.eig(Power_f.dot(Power_f.T))
idx = vals.argsort()[::-1]
vals = vals[idx]
vecs = vecs[:, idx]
vals_num = np.copy(vals/np.sum(vals))
EOFs = []
PCs = []
for i in range(len(Power_f)-1):
    EOFs.append(vecs[:, i])
    PCs.append(vecs[:, i].dot(Power_f))

# --------------------------------------------------------------------- #
# Savings and rescaling
# --------------------------------------------------------------------- #

pd.DataFrame(PCs).to_pickle(PATH_PROCDATA+'PCs/'+ex+'_fs.pkl')
pd.DataFrame(EOFs).to_pickle(PATH_PROCDATA+'EOFs/'+ex+'_fs.pkl')
pd.DataFrame(vals_num).to_pickle(PATH_PROCDATA+'EVs/'+ex+'_fs.pkl')
