# --------------------------------------------------------------------- #
# Preambule
# --------------------------------------------------------------------- #

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib as mpl
import sys
import matplotlib.pyplot as plt
from scipy.fftpack import rfftfreq, rfft
import scipy.io
import os
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
dir_data = '/Users/mmdekker/Documents/Werk/Data/Braindata/parLFP.mat'

def windowmean(Data,size):
    if size==1: return Data
    elif size==0:
        print('Size 0 not possible!')
        sys.exit()
    else:
        Result=np.zeros(len(Data))+np.nan
        for i in range(np.int(size/2.),np.int(len(Data)-size/2.)):
            Result[i]=np.nanmean(Data[i-np.int(size/2.):i+np.int(size/2.)])
        return np.array(Result)
    
def RemovePowerlaw(Data,freq):
    def fc(x, a, b):
        return x**a * b
    Hz1 = np.where(freq>1)[0][0] # only from 1 Hz and larger
    Hz2 = np.where(freq<=200)[0][-1] # only from 200 Hz and lower
    frq = freq[Hz1:Hz2]
    data = Data[Hz1:Hz2]
    popt, pcov = curve_fit(fc, frq, data)
    residual = data - fc(frq,popt[0],popt[1])
    return np.array(residual),np.array(frq)

# --------------------------------------------------------------------- #
# Read data
# --------------------------------------------------------------------- #

if not 'mat' in locals(): mat = scipy.io.loadmat('/Users/mmdekker/Documents/Werk/Data/Braindata/parLFP.mat')
Data = mat['EEG'][0][0]
Time = Data[14].T
Extra = Data[-1]
Series = np.array(Data[15])
pd.DataFrame(Series).to_pickle('/Users/mmdekker/Documents/Werk/Scripts/__Other/BrainProject/Mike/ProcData/Series_nn.pkl')

Series2 = Series.T - Series.mean(axis=1)
stds = Series2.std(axis=0)
stds[stds==0] = 1
Series2 = Series2/stds
Series2 = Series2.T
pd.DataFrame(Series2).to_pickle('/Users/mmdekker/Documents/Werk/Scripts/__Other/BrainProject/Mike/ProcData/Series.pkl')

ffts = []
ffts_wm = []
for i in tqdm(range(len(Series2)),file=sys.stdout):
    FFT = np.abs(rfft(Series2[i]))/len(Series2[i])*2
    ffts.append(FFT)
    ffts_wm.append(windowmean(FFT,50))
ffts_wm = np.array(ffts_wm)
ffts_wm = np.array(ffts_wm)[:,25:]
ffts_wm = np.array(ffts_wm)[:,:-25]
freq = rfftfreq(len(ffts_wm[0]),d=1e-3)
ffts_wm = (ffts_wm.T/np.sum(ffts_wm,axis=1)).T

pd.DataFrame(ffts).to_pickle('/Users/mmdekker/Documents/Werk/Scripts/__Other/BrainProject/Mike/ProcData/FFTs.pkl')
pd.DataFrame(ffts_wm).to_pickle('/Users/mmdekker/Documents/Werk/Scripts/__Other/BrainProject/Mike/ProcData/FFTs_wm.pkl')


Hz1 = np.where(freq>1)[0][0] # only from 1 Hz and larger
Hz2 = np.where(freq<=200)[0][-1] # only from 200 Hz and lower
frq = freq[Hz1:Hz2]
resids = []
for i in tqdm(range(len(ffts_wm)),file=sys.stdout):
    a,b = RemovePowerlaw(ffts_wm[i],freq)
    resids.append(a)
    frq = b

pd.DataFrame(resids).to_pickle('/Users/mmdekker/Documents/Werk/Scripts/__Other/BrainProject/Mike/ProcData/Resids.pkl')
