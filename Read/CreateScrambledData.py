# --------------------------------------------------------------------- #
# Create scrambled data
# --------------------------------------------------------------------- #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
datadir = '/Users/mmdekker/Documents/Werk/Data/Braindata/ObjectTest.mat'
import random

# --------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------- #

ex      = 'D_PFC'
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
# Scramble data
# --------------------------------------------------------------------- #

inds = np.arange(0,len(Power[0]))
random.shuffle(inds)
Power_s = Power[:,inds]

# --------------------------------------------------------------------- #
# Project onto EOFs
# --------------------------------------------------------------------- #

PCs_s = []
PCs_r = []
for i in range(len(Power)-1):
    PCs_s.append(EOFs[i].dot(Power_s))
    PCs_r.append(EOFs[i].dot(Power))

pd.DataFrame(PCs_s).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/PCs/'+ex+'_s.pkl')
