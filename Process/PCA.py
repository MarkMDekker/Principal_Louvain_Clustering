## --------------------------------------------------------------------- #
## Principal Component Analysis
## --------------------------------------------------------------------- #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
datadir = '/Users/mmdekker/Documents/Werk/Data/Braindata/ObjectTest.mat'

## --------------------------------------------------------------------- #
## Read data
## --------------------------------------------------------------------- #

ex = 'HR_HIP'
ProcessedPower = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/ProcessedPower/'+ex+'.pkl'))
ProcessedPower_red = ProcessedPower#[:50]
Steps = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/Steps/'+ex+'.pkl')).T[0]
Synchronization = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/Synchronization/'+ex+'.pkl')).T[0]

## --------------------------------------------------------------------- #
## Select data
## --------------------------------------------------------------------- #

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

#%%
## --------------------------------------------------------------------- #
## Full PCA calculations
## --------------------------------------------------------------------- #

vals,vecs = np.linalg.eig(ProcessedPower_red.dot(ProcessedPower_red.T))
idx = vals.argsort()[::-1]
vals = vals[idx]
vecs = vecs[:,idx]
vals_num = np.copy(vals/np.sum(vals))
EOFs = []
PCs = []
for i in range(len(ProcessedPower_red)-1):
    EOFs.append(vecs[:,i])
    PCs.append(vecs[:,i].dot(ProcessedPower_red))

## --------------------------------------------------------------------- #
## Savings and rescaling
## --------------------------------------------------------------------- #

pd.DataFrame(PCs).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/PCs/'+ex+'.pkl')
pd.DataFrame(EOFs).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/EOFs/'+ex+'.pkl')
pd.DataFrame(vals_num).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/EVs/'+ex+'.pkl')
#pd.DataFrame(PCs_spec).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ObjectExperiment2/PC_compsspec.pkl')
#PCs_all_spec=pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ObjectExperiment2/PC_compsall.pkl')

## --------------------------------------------------------------------- #
## Old code
## --------------------------------------------------------------------- #

# Rescale Synchronization
#Mvec = np.copy(Synchronization)
#Mvec = Mvec - np.min(Mvec)
#Mvec = Mvec / np.max(Mvec)
#vals,vecs = np.linalg.eig(ProcessedPower_red.dot(ProcessedPower_red.T))
#idx = vals.argsort()[::-1]
#vals = vals[idx]
#vecs = vecs[:,idx]
#vals_num = np.copy(vals/np.sum(vals))
#EOFs = []
#PCs = []
#for i in range(len(ProcessedPower_red)):
#    EOFs.append(vecs[:,i])
#    PCs.append(vecs[:,i].dot(ProcessedPower_red))
#
#PCs_spec = [PCs]
#PCs_all_spec = [PCs]
#EOFs_spec = [EOFs]
#vals_nums_spec = [vals_num]
#
#for i in range(len(mins)):
#    PP = ProcessedPower_red[:,(Steps/1000.>=mins[i]) & (Steps/1000.<=maxs[i])]
#    vals,vecs = np.linalg.eig(PP.dot(PP.T))
#    idx = vals.argsort()[::-1]
#    vals = vals[idx]
#    vals_num_i = np.copy(vals/np.sum(vals))
#    vecs = vecs[:,idx]
#    EOFs_i = []
#    PCs_i = []
#    for i in range(len(PP)):
#        EOFs_i.append(vecs[:,i])
#        PCs_i.append(vecs[:,i].dot(PP))
#    PCs_spec.append(np.copy(PCs_i))
#    EOFs_spec.append(np.copy(EOFs_i))
#    vals_nums_spec.append(np.copy(vals_num_i))
#    PCs_full = []
#    for i in range(len(ProcessedPower_red)):
#        PCs_full.append(EOFs_i[i].dot(ProcessedPower_red))
#    PCs_all_spec.append(PCs_full)
#
### --------------------------------------------------------------------- #
### Savings and rescaling
### --------------------------------------------------------------------- #
#
#pd.DataFrame(PCs_all_spec).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/PCs/'+ex+'.pkl')
##pd.DataFrame(EOFs_spec).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ObjectExperiment2/PC_eofs.pkl')
#pd.DataFrame(vals_nums_spec).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/EVs/'+ex+'.pkl')
##pd.DataFrame(PCs_spec).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ObjectExperiment2/PC_compsspec.pkl')
#
##PCs_all_spec=pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ObjectExperiment2/PC_compsall.pkl')
#
## Rescale Synchronization
#Mvec = np.copy(Synchronization)
#Mvec = Mvec - np.min(Mvec)
#Mvec = Mvec / np.max(Mvec)