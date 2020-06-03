# --------------------------------------------------------------------------- #
# Read datafile ObjectTest.mat from Mike
# Rodent exploring old/new objects after a training phase
# 5 min box 1
# 10 min exploring
# 5 min box 2
# 60 min in between
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Preambule
# --------------------------------------------------------------------------- #

# Standard packages
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.io
import os

# Packages for functions
from scipy.fftpack import rfftfreq, rfft
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
import scipy.io
import warnings
warnings.filterwarnings("ignore")
os.chdir('/Users/mmdekker/Documents/Werk/Scripts/Projects_Side/'
         'PhaseSpaces_In_Brain/Read')
from Functions import MeanCorrelation, Abundances, Standardize, FourierTrans
from Functions import Powerlaw, ProcessPower

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

figdir = '/Users/mmdekker/Documents/Werk/Figures/Neurostuff/Dataset_new/'
path_pdata = ('/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/'
              'ProcessedData/')
st = ['all', 'nexp', 'exp']
stC = ['All', 'Nexp', 'Exp']

# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #

N = 600000  # Amount of data points per signal
prt = 0  # part of the dataset: for confidence intervals
f_low = 1  # frequency lower bound
f_high = 100  # frequency upper bound
save = 1

# --------------------------------------------------------------------------- #
# Data reading
# --------------------------------------------------------------------------- #

mat = scipy.io.loadmat(path_pdata+'../ObjectTest.mat')
DataAll = mat['EEG'][0][0]
Time = DataAll[14]
Data_dat = DataAll[15][:, N*prt:N*prt+N]
Data_re1 = DataAll[17]
Data_re2 = DataAll[18]
Data_re3 = DataAll[19]
Data_re4 = DataAll[25]
Data_re5 = DataAll[30]
Data_re6 = DataAll[39]
Data_vid = DataAll[40].T
Data_vid = Data_vid[:, N*prt:N*prt+N]
Data_ran = np.arange(len(Data_vid[0]))

# --------------------------------------------------------------------------- #
# Filter exploration
# Used acronyms: A = exploring new, B = exploring old, C = not exploring
# --------------------------------------------------------------------------- #

Exploring = Data_ran[(Data_vid[2] == 1) | (Data_vid[3] == 1)]
NotExploring = Data_ran[(Data_vid[2] != 1) & (Data_vid[3] != 1)]
D_all = Data_dat
D_nexp = Data_dat[:, NotExploring]
D_exp = Data_dat[:, Exploring]

# --------------------------------------------------------------------------- #
# Standardizing data sets: not necessary?
# --------------------------------------------------------------------------- #

SD_all = Standardize(D_all)
SD_nexp = Standardize(D_nexp)
SD_exp = Standardize(D_exp)

# --------------------------------------------------------------------------- #
# Fourier transform
# --------------------------------------------------------------------------- #

Ds = [D_all, D_nexp, D_exp]
for i in range(3):

    # Perform Fourier Transformations
    F, Fwm, freq = FourierTrans(Ds[i])

    # Remove power laws
    R, freq2 = Powerlaw(Fwm, freq, f_low, f_high)

    if save == 1:
        pd.DataFrame(F).to_pickle(path_pdata+'FFTs/'+stC[i]+'/'+str(N) +
                                  '_'+str(prt)+'.pkl')
        pd.DataFrame(Fwm).to_pickle((path_pdata+'FFTs/'+stC[i]+'/'+str(N) +
                                     '_'+str(prt)+'_wm.pkl'))
        pd.DataFrame(R).to_pickle((path_pdata+'Resids/'+stC[i]+'/'+str(N) +
                                   '_'+str(prt)+'.pkl'))
        pd.DataFrame(freq2).to_pickle(path_pdata+'Resids/Freq.pkl')

# --------------------------------------------------------------------- #
# Perform brain-specific analyses
# => On the exploration part (D)
# --------------------------------------------------------------------- #

# --------------------------------------------------------------------- #
# PFC
# --------------------------------------------------------------------- #

Sets = [range(16), range(16, 24), range(24, 32)]
Strs = ['PFC', 'PAR', 'HIP']
for i in range(3):
    Sync, RelP, St = MeanCorrelation(SD_all, 500, N, f_low, f_high, Sets[i])
    Proc = ProcessPower(RelP)
    pd.DataFrame(Sync).to_pickle(path_pdata+'Synchronization/'+Strs[i]+'/'+str(N) +
                                  '_'+str(prt)+'.pkl')
    pd.DataFrame(RelP).to_pickle(path_pdata+'RelativePower/'+Strs[i]+'/'+str(N) +
                                  '_'+str(prt)+'.pkl')
    pd.DataFrame(St).to_pickle(path_pdata+'Steps/'+Strs[i]+'/'+str(N) +
                               '_'+str(prt)+'.pkl')
    pd.DataFrame(Proc).to_pickle(path_pdata+'ProcessedPower/'+Strs[i]+'/'+str(N) +
                                  '_'+str(prt)+'.pkl')


#%%

# --------------------------------------------------------------------- #
# Process new, old and exploring vectors
# --------------------------------------------------------------------- #

Stepjes = np.arange(0,600000,1)
New = np.zeros(len(Stepjes))
Old = np.zeros(len(Stepjes))
Exp = np.zeros(len(Stepjes))
for i in tqdm(range(len(ExploringNew))):
    which = np.where(np.abs(ExploringNew[i]-Steps_HIP)==np.nanmin(np.abs(ExploringNew[i]-Steps_HIP)))[0][0]
    New[which] = 1
for i in tqdm(range(len(ExploringOld))):
    which = np.where(np.abs(ExploringOld[i]-Steps_HIP)==np.nanmin(np.abs(ExploringOld[i]-Steps_HIP)))[0][0]
    Old[which] = 1
Exp[Old==1] = 1
Exp[New==1] = 1

# make exploration continuous
Newcont = windowmean(New,100)
Oldcont = windowmean(Old,100)
Expcont = windowmean(Exp,100)

pd.DataFrame(Newcont).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/NewOld/Newcont_hr.pkl')
pd.DataFrame(Oldcont).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/NewOld/Oldcont_hr.pkl')
pd.DataFrame(Expcont).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/NewOld/Expcont_hr.pkl')
pd.DataFrame(New).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/NewOld/New_hr.pkl')
pd.DataFrame(Old).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/NewOld/Old_hr.pkl')
pd.DataFrame(Exp).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/NewOld/Exp_hr.pkl')
