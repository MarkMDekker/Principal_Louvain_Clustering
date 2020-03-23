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
from Functions import Powerlaw

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

N = 25000  # Amount of data points per signal
f_low = 1  # frequency lower bound
f_high = 100  # frequency upper bound

# --------------------------------------------------------------------------- #
# Data reading
# --------------------------------------------------------------------------- #

mat = scipy.io.loadmat(path_pdata+'../ObjectTest.mat')
DataAll = mat['EEG'][0][0]
Time = DataAll[14]
Data_dat = DataAll[15][:, :N]
Data_re1 = DataAll[17]
Data_re2 = DataAll[18]
Data_re3 = DataAll[19]
Data_re4 = DataAll[25]
Data_re5 = DataAll[30]
Data_re6 = DataAll[39]
Data_vid = DataAll[40].T
Data_vid = Data_vid[:, :N]

# --------------------------------------------------------------------------- #
# Filter exploration
# Used acronyms: A = exploring new, B = exploring old, C = not exploring
# --------------------------------------------------------------------------- #

ExploringNew = np.where(Data_vid[2] == 1)[0]
ExploringOld = np.where(Data_vid[3] == 1)[0]
Exploring = np.unique(list(ExploringNew)+list(ExploringOld))
NotExploring = np.unique(np.array(list(np.where(Data_vid[2] != 1)[0]) +
                                  list(np.where(Data_vid[3] != 1)[0])))
D_all = Data_dat
D_nexp = Data_dat[:, NotExploring]
D_exp = Data_dat[:, Exploring]

# --------------------------------------------------------------------------- #
# Standardizing data sets: not necessary?
# --------------------------------------------------------------------------- #

#SD_all = Standardize(D_all)
#SD_nexp = Standardize(D_nexp)
#SD_exp = Standardize(D_exp)

# --------------------------------------------------------------------------- #
# Fourier transform
# --------------------------------------------------------------------------- #

Ds = [D_all, D_nexp, D_exp]
for i in range(3):

    # Fourier Transform
    F, Fwm, freq = FourierTrans(Ds[i])
    pd.DataFrame(F).to_pickle(path_pdata+'FFTs/'+stC[i]+'/'+str(N)+'.pkl')
    pd.DataFrame(Fwm).to_pickle(path_pdata+'FFTs/'+stC[i]+'/'+str(N)+'_wm.pkl')

    # Remove power laws
    R, freq2 = Powerlaw(Fwm, freq, f_low, f_high)
    pd.DataFrame(R).to_pickle(path_pdata+'Resids/'+stC[i]+'/'+str(N)+'.pkl')
    pd.DataFrame(freq2).to_pickle(path_pdata+'Resids/Freq.pkl')

#%%
FourierTrans(Data_B,'B')
FourierTrans(Data_C,'C')
FourierTrans(Data_D,'D')

F_A     = pd.read_pickle(path_pdata+'FFTs/FFTs_A.pkl')
Fwm_A   = pd.read_pickle(path_pdata+'FFTs/FFTs_wm_A.pkl')
F_B     = pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/FFTs/FFTs_B.pkl')
Fwm_B   = pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/FFTs/FFTs_wm_B.pkl')
F_C     = pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/FFTs/FFTs_C.pkl')
Fwm_C   = pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/FFTs/FFTs_wm_C.pkl')
F_D     = pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/FFTs/FFTs_D.pkl')
Fwm_D   = pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/FFTs/FFTs_wm_D.pkl')

Fwm_A = np.array(Fwm_A)[:,25:]
Fwm_A = np.array(Fwm_A)[:,:-25]
Fwm_B = np.array(Fwm_B)[:,25:]
Fwm_B = np.array(Fwm_B)[:,:-25]
Fwm_C = np.array(Fwm_C)[:,25:]
Fwm_C = np.array(Fwm_C)[:,:-25]
Fwm_D = np.array(Fwm_D)[:,25:]
Fwm_D = np.array(Fwm_D)[:,:-25]

freq = rfftfreq(len(Fwm_A[0]),d=1e-3)
Fwm_A = (Fwm_A.T/np.sum(Fwm_A,axis=1)).T
Fwm_B = (Fwm_B.T/np.sum(Fwm_B,axis=1)).T
Fwm_C = (Fwm_C.T/np.sum(Fwm_C,axis=1)).T
Fwm_D = (Fwm_D.T/np.sum(Fwm_D,axis=1)).T

# --------------------------------------------------------------------- #
# Remove power laws
# --------------------------------------------------------------------- #

#Powerlaw(Fwm_A,'A',freq)
#Powerlaw(Fwm_B,'B',freq)
#Powerlaw(Fwm_C,'C',freq)
#Powerlaw(Fwm_D,'D',freq)

rA = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/Resids/Resids_A.pkl'))
rB = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/Resids/Resids_B.pkl'))
rC = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/Resids/Resids_C.pkl'))
rD = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/Resids/Resids_D.pkl'))
frq = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/Resids/Resids_frq.pkl'))

# --------------------------------------------------------------------- #
# Calculate covariance matrices (overall)
# --------------------------------------------------------------------- #

#CovMatA = Std_A.dot(Std_A.T)
#CovMatB = Std_B.dot(Std_B.T)
#CovMatC = Std_C.dot(Std_C.T)
#CovMatD = Std_D.dot(Std_D.T)
#pd.DataFrame(CovMatA/np.nanmax(np.abs(CovMatA))).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/CovMat/CovMatA.pkl')
#pd.DataFrame(CovMatB/np.nanmax(np.abs(CovMatB))).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/CovMat/CovMatB.pkl')
#pd.DataFrame(CovMatC/np.nanmax(np.abs(CovMatC))).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/CovMat/CovMatC.pkl')
#pd.DataFrame(CovMatD/np.nanmax(np.abs(CovMatD))).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/CovMat/CovMatD.pkl')

CovMatA = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/CovMat/CovMatA.pkl'))
CovMatB = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/CovMat/CovMatB.pkl'))
CovMatC = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/CovMat/CovMatC.pkl'))
CovMatD = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/CovMat/CovMatD.pkl'))

# --------------------------------------------------------------------- #
# Calculate correlation matrices (overall, residues)
# --------------------------------------------------------------------- #

#CorrMatA = np.corrcoef(rA)
#CorrMatB = np.corrcoef(rB)
#CorrMatC = np.corrcoef(rC)
#CorrMatD = np.corrcoef(rD)
#pd.DataFrame(CorrMatA).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/CorrMat/CorrMatA.pkl')
#pd.DataFrame(CorrMatB).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/CorrMat/CorrMatB.pkl')
#pd.DataFrame(CorrMatC).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/CorrMat/CorrMatC.pkl')
#pd.DataFrame(CorrMatD).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/CorrMat/CorrMatD.pkl')

CorrMatA = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/CorrMat/CorrMatA.pkl'))
CorrMatB = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/CorrMat/CorrMatB.pkl'))
CorrMatC = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/CorrMat/CorrMatC.pkl'))
CorrMatD = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/CorrMat/CorrMatD.pkl'))

#%%
# --------------------------------------------------------------------- #
# Perform brain-specific analyses
# => On the exploration part (D)
# --------------------------------------------------------------------- #

# --------------------------------------------------------------------- #
# PFC
# --------------------------------------------------------------------- #

Synchronization_PFC, RelativePower_PFC, Steps_PFC = MeanCorrelation(Std_D,500,200000,range(16))
ProcessedPower_PFC = ProcessPower(RelativePower_PFC)
pd.DataFrame(Synchronization_PFC).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/Synchronization/HR_PFC.pkl')
pd.DataFrame(RelativePower_PFC).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/RelativePower/HR_PFC.pkl')
pd.DataFrame(Steps_PFC).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/Steps/HR_PFC.pkl')
pd.DataFrame(ProcessedPower_PFC).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/ProcessedPower/HR_PFC.pkl')
#%%
# --------------------------------------------------------------------- #
# HIP
# --------------------------------------------------------------------- #

Synchronization_HIP, RelativePower_HIP, Steps_HIP = MeanCorrelation(Std_D,500,200000,range(24,32))
ProcessedPower_HIP = ProcessPower(RelativePower_HIP)
pd.DataFrame(Synchronization_HIP).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/Synchronization/HR_HIP.pkl')
pd.DataFrame(RelativePower_HIP).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/RelativePower/HR_HIP.pkl')
pd.DataFrame(Steps_HIP).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/Steps/HR_HIP.pkl')
pd.DataFrame(ProcessedPower_HIP).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/ProcessedPower/HR_HIP.pkl')
#%%
# --------------------------------------------------------------------- #
# PAR
# --------------------------------------------------------------------- #

Synchronization_PAR, RelativePower_PAR, Steps_PAR = MeanCorrelation(Std_D,500,200000,range(16,24))
ProcessedPower_PAR = ProcessPower(RelativePower_PAR)
pd.DataFrame(Synchronization_PAR).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/Synchronization/HR_PAR.pkl')
pd.DataFrame(RelativePower_PAR).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/RelativePower/HR_PAR.pkl')
pd.DataFrame(Steps_PAR).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/Steps/HR_PAR.pkl')
pd.DataFrame(ProcessedPower_PAR).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/ProcessedPower/HR_PAR.pkl')

#%%
# --------------------------------------------------------------------- #
# Perform brain-specific analyses
# => On all data
# --------------------------------------------------------------------- #

# --------------------------------------------------------------------- #
# PFC
# --------------------------------------------------------------------- #

Synchronization_PFC, RelativePower_PFC, Steps_PFC = MeanCorrelation(Std_all,500,500000,range(16))
ProcessedPower_PFC = ProcessPower(RelativePower_PFC)
pd.DataFrame(Synchronization_PFC).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/Synchronization/PFC_hr.pkl')
pd.DataFrame(RelativePower_PFC).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/RelativePower/PFC_hr.pkl')
pd.DataFrame(Steps_PFC).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/Steps/PFC_hr.pkl')
pd.DataFrame(ProcessedPower_PFC).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/ProcessedPower/PFC_hr.pkl')
#%%
# --------------------------------------------------------------------- #
# HIP
# --------------------------------------------------------------------- #

Synchronization_PAR, RelativePower_PAR, Steps_PAR = MeanCorrelation(Std_all,500,500000,range(16,24))
ProcessedPower_PAR = ProcessPower(RelativePower_PAR)
pd.DataFrame(Synchronization_PAR).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/Synchronization/PAR_hr.pkl')
pd.DataFrame(RelativePower_PAR).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/RelativePower/PAR_hr.pkl')
pd.DataFrame(Steps_PAR).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/Steps/PAR_hr.pkl')
pd.DataFrame(ProcessedPower_PAR).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/ProcessedPower/PAR_hr.pkl')
#%%
# --------------------------------------------------------------------- #
# PAR
# --------------------------------------------------------------------- #

Synchronization_HIP, RelativePower_HIP, Steps_HIP = MeanCorrelation(Std_all,500,500000,range(24,32))
ProcessedPower_HIP = ProcessPower(RelativePower_HIP)
pd.DataFrame(Synchronization_HIP).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/Synchronization/HIP_hr.pkl')
pd.DataFrame(RelativePower_HIP).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/RelativePower/HIP_hr.pkl')
pd.DataFrame(Steps_HIP).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/Steps/HIP_hr.pkl')
pd.DataFrame(ProcessedPower_HIP).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/ProcessedPower/HIP_hr.pkl')
#%%
# --------------------------------------------------------------------- #
# All brain regions
# --------------------------------------------------------------------- #

Synchronization, RelativePower, Steps = MeanCorrelation(Std_all,500,100000,range(32))
ProcessedPower = ProcessPower(RelativePower)
pd.DataFrame(Synchronization).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/Synchronization/ALL_hr1.pkl')
pd.DataFrame(RelativePower).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/RelativePower/ALL_hr1.pkl')
pd.DataFrame(Steps).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/Steps/ALL_hr1.pkl')
pd.DataFrame(ProcessedPower).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/ProcessedPower/ALL_hr1.pkl')
##%%
#Synchronization = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ObjectExperiment/Synchronization.pkl')).T[0]
#RelativePower = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ObjectExperiment/RelativePower.pkl'))
#Steps = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ObjectExperiment/Steps.pkl')).T[0]
#ProcessedPower = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ObjectExperiment/ProcessedPower.pkl'))

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
