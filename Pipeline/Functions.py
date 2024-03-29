# ----------------------------------------------------------------- #
# About script
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from scipy.fftpack import rfftfreq, rfft
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------- #
# Functions
# ----------------------------------------------------------------- #


def Standardize(Data):
    '''
    Function to standardize dataset.
    '''
    Data = Data.T
    Data = Data.T - np.nanmean(Data, axis=1)
    stds = np.nanstd(Data, axis=0)
    stds[stds == 0] = 1
    Data = Data/stds
    return Data


def windowmean(Data, size):
    if size == 1:
        return Data
    elif size == 0:
        print('Size 0 not possible!')
        sys.exit()
    else:
        Result = np.zeros(len(Data))+np.nan
        for i in range(np.int(size/2.), np.int(len(Data)-size/2.)):
            Result[i] = np.nanmean(Data[i-np.int(size/2.):i+np.int(size/2.)])
        return np.array(Result)


def FourierTrans(Data):
    '''
    Take Fourier Transform of whole set,
    for every signal in the data.
    '''
    ffts = []
    ffts_wm = []
    for i in tqdm(range(len(Data))):
        FFT = np.abs(rfft(Data[i]))/len(Data[i])*2
        ffts.append(FFT)
        ffts_wm.append(windowmean(FFT, 25))
        if i == 0:
            freq = rfftfreq(len(FFT), d=1e-3)
    return ffts, ffts_wm, freq


def Abundances(Data, frq, borders, wheres):
    abund = []
    for i in range(len(borders)-1):
        abund.append(np.mean(Data[wheres[i]]))
    return abund


def fc(x, a, b):
    return x**a * b


def fc2(x, a, b, c):
    return x**a * b + c


def RemovePowerlaw2(Data, frq, Hz1, Hz2):
    data = Data[Hz1:Hz2]
    try:
        popt, pcov = curve_fit(fc, frq, data, p0=[-0.5, 0.02])
    except:
        popt = [np.nan, np.nan]
    residual = data - fc(frq, popt[0], popt[1])
    return np.array(residual)



def RemovePowerlaw2a(Data, frq, Hz1, Hz2):
    data = Data[Hz1:Hz2]
    try:
        popt, pcov = curve_fit(fc, frq, data, p0=[-0.5, 0.02])
    except:
        popt = [np.nan, np.nan]
    residual = data - fc(frq, popt[0], popt[1])
    return popt[0], popt[1], np.array(residual)



def RemovePowerlaw2b(Data, frq, Hz1, Hz2):
    data = Data[Hz1:Hz2]
    try:
        popt, pcov = curve_fit(fc, frq[1:], data[1:], p0=[-1, 0.02], maxfev=2000)
    except:
        popt = [np.nan, np.nan, np.nan]
    residual = data - fc(frq, popt[0], popt[1])
    return popt[0], popt[1], np.array(residual)



def RemovePowerlaw3(Data, frq, Hz1, Hz2):
    data = Data[Hz1:Hz2]
    try:
        popt, pcov = curve_fit(fc2, frq[1:], data[1:], p0=[-0.5, 0.02, 0], maxfev=2000)
    except:
        popt = [np.nan, np.nan, np.nan]
    residual = data - fc2(frq, popt[0], popt[1], popt[2])
    return popt[0], popt[1], popt[2], residual


def ProcessPower(RelativePower):
    Power_proc = []
    for i in range(len(RelativePower[0])):
        Sery = np.copy(RelativePower[:, i])
        Sery = Sery - np.nanmean(Sery)
        Sery = Sery / np.nanstd(Sery)
        Power_proc.append(Sery)
    return np.array(Power_proc)


def func12(vec):
    mat = np.zeros(shape=(res2, res2))
    a = 0
    for i in range(res2):
        for j in range(res2):
            mat[i, j] = vec[a]
            a += 1
    return mat


def func21(mat):
    return mat.flatten()


def overlap_metric(Cm, Cc):
    exism = np.copy(Cm)
    exisc = np.copy(Cc)
    intersect = np.intersect1d(exisc, exism)
    score = len(intersect) / (1e-9+len(exisc))    
    return score