# --------------------------------------------------------------------- #
# Create scrambled data
# --------------------------------------------------------------------- #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
datadir = '/Users/mmdekker/Documents/Werk/Data/Braindata/ObjectTest.mat'
import random
from tqdm import tqdm

# --------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------- #

ex      = 'HR_PAR'
save    = 0

# --------------------------------------------------------------------- #
# Read data
# --------------------------------------------------------------------- #

Power               = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/ProcessedPower/'+ex+'.pkl'))
Steps               = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/Steps/'+ex+'.pkl')).T[0]
Synchronization     = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/Synchronization/'+ex+'.pkl')).T[0]
PCs                 = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/PCs/'+ex+'.pkl'))
EOFs                = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/EOFs/'+ex+'.pkl'))

# --------------------------------------------------------------------- #
# Select data
# --------------------------------------------------------------------- #

mat = scipy.io.loadmat(datadir)
DataAll = mat['EEG'][0][0]
Time        = DataAll[14]
Data_dat    = DataAll[15]
Data_re1    = DataAll[17]
Data_re2    = DataAll[18]
Data_re3    = DataAll[19]
Data_re4    = DataAll[25]
Data_re5    = DataAll[30]
Data_re6    = DataAll[39]
Data_vid    = DataAll[40].T
ExploringNew = np.where(Data_vid[2] == 1)[0]
ExploringOld = np.where(Data_vid[3] == 1)[0]
NotExploring = np.array(list(np.where(Data_vid[2] != 1)[0]) + list(np.where(Data_vid[3] != 1)[0]))

# --------------------------------------------------------------------- #
# Cut and swap data
# --------------------------------------------------------------------- #

inds = np.arange(0,len(Power[0]))
Power_c = np.copy(Power)
for k in tqdm(range(1000)):
    ind = np.random.choice(inds)
    inds_new = list(inds[ind:])+list(inds[:ind])
    Power_c = Power_c[:,inds_new]
#Power = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/ProcessedPower/'+ex+'_cs.pkl'))

## --------------------------------------------------------------------- #
## Full PCA calculations
## --------------------------------------------------------------------- #

vals,vecs = np.linalg.eig(Power_c.dot(Power_c.T))
idx = vals.argsort()[::-1]
vals = vals[idx]
vecs = vecs[:,idx]
vals_num = np.copy(vals/np.sum(vals))
EOFs = []
PCs = []
for i in range(len(Power_c)-1):
    EOFs.append(vecs[:,i])
    PCs.append(vecs[:,i].dot(Power_c))

## --------------------------------------------------------------------- #
## Savings and rescaling
## --------------------------------------------------------------------- #

pd.DataFrame(PCs).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/PCs/'+ex+'_cs.pkl')
pd.DataFrame(EOFs).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/EOFs/'+ex+'_cs.pkl')
pd.DataFrame(vals_num).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/EVs/'+ex+'_cs.pkl')