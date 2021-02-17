# =========================================================================== #
# Functions for the brain project
# =========================================================================== #

import numpy as np
from tqdm import tqdm
import sys
from scipy.fftpack import rfftfreq, rfft
from scipy.optimize import curve_fit

# =========================================================================== #
# Pre-process functions
# =========================================================================== #


def Standardize(Data):
    '''
    Function to standardize dataset.
    Not being used right now.
    '''
    Data = Data.T
    Data = Data.T - Data.mean(axis=1)
    stds = Data.std(axis=0)
    stds[stds == 0] = 1
    Data = Data/stds
    return Data


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


def Powerlaw(ffts_wm, freq, f_low, f_high):
    '''
    Iterate over signals, look at the power spectrum, fit powerlaw and
    remove this power law from the data. Returns residuals.
    '''
    resids = []
    for i in tqdm(range(len(ffts_wm))):
        a, b = RemovePowerlaw(ffts_wm[i], freq, f_low, f_high)
        resids.append(a)
        frq = b
    return resids, frq


def fc(x, a, b):
    return x**a * b


def RemovePowerlaw(Data, freq, f_low, f_high):
    Hz1 = np.where(freq > f_low)[0][0]
    Hza = np.where(~np.isnan(Data))[0][0]
    Hz1 = np.max([Hz1, Hza])
    Hz2 = np.where(freq <= f_high)[0][-1]
    frq = freq[Hz1:Hz2]
    data = Data[Hz1:Hz2]
    popt, pcov = curve_fit(fc, frq, data)
    residual = data - fc(frq, popt[0], popt[1])
    return np.array(residual), np.array(frq)


# =========================================================================== #
# Core function
# =========================================================================== #


def MeanCorrelation(Data, ws, N, f_low, f_high, wh=[]):
    N = np.min([N, len(Data[0])])
    if len(wh) == 0:
        wh = range(len(Data))
    abundsa = []
    means = []
    steps = np.arange(ws, N-ws)
    sers = Data[:, ws-ws:ws+ws]
    freq_red = rfftfreq(len(sers[0]), d=1e-3)

    Hz1 = np.where(freq_red >= f_low-0.5)[0][0]
    Hz2 = np.where(freq_red <= f_high+0.5)[0][-1]
    frq = freq_red[Hz1:Hz2]

    borders = np.arange(0, 101, 2)
    wheres = []
    ran = np.arange(len(frq))
    for i in range(len(borders)-1):
        wheres.append(ran[(frq > borders[i]) & (frq <= borders[i+1])])

    for step in tqdm(steps):
        ffts = []
        sers = Data[wh, step-ws:step+ws]
        abunds = []
        for i in range(len(sers)):
            FFT = np.abs(rfft(sers[i]))/len(sers[i])*2
            FFT = FFT[~np.isnan(FFT)]/np.sum(FFT)
            FFTn = RemovePowerlaw2(FFT, frq, Hz1, Hz2)
            ffts.append(FFTn)
            abunds.append(Abundances(FFTn, frq, borders, wheres))
        abundsa.append(np.nanmean(abunds, axis=0))
        means.append(np.nanmean(np.corrcoef(ffts)))
    return np.array(means), np.array(abundsa), np.array(steps)


def Abundances(Data, frq, borders, wheres):
    abund = []
    for i in range(len(borders)-1):
        abund.append(np.mean(Data[wheres[i]]))
    return abund


def RemovePowerlaw2(Data, frq, Hz1, Hz2):
    data = Data[Hz1:Hz2]
    try:
        popt, pcov = curve_fit(fc, frq, data, p0=[-0.5, 0.02])
    except:
        popt = [np.nan, np.nan]
    ind = 0
    while len(frq) != len(data) and ind < 100:
        data = np.array(list(data)+[0])
        ind += 1
    residual = data - fc(frq, popt[0], popt[1])
    return np.array(residual)


# =========================================================================== #
# Helper functions
# =========================================================================== #


def fc(x, a, b):
    return x**a * b

def WeightFreq(data):
    FFT = rfft(data)
    freq = rfftfreq(len(data),d=1)
    return np.dot(FFT,freq)/len(FFT)

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
    
def windowfreq(Data,size):
    if size==1: return WeightFreq(Data)
    elif size==0:
        print('Size 0 not possible!')
        sys.exit()
    else:
        Result=np.zeros(len(Data))+np.nan
        for i in range(np.int(size/2.),np.int(len(Data)-size/2.)):
            Result[i]=WeightFreq(Data[i-np.int(size/2.):i+np.int(size/2.)])
        return np.array(Result)

def relabund(fft,freq):
    freq_t = np.sum(fft[np.intersect1d(np.where(freq>=3)[0],np.where(freq<=8)[0])])/np.sum(fft)
    freq_a = np.sum(fft[np.intersect1d(np.where(freq>=8)[0],np.where(freq<=12)[0])])/np.sum(fft)
    freq_b = np.sum(fft[np.intersect1d(np.where(freq>=12)[0],np.where(freq<=38)[0])])/np.sum(fft)
    freq_g = np.sum(fft[np.intersect1d(np.where(freq>=38)[0],np.where(freq<=42)[0])])/np.sum(fft)
    return [freq_t,freq_a,freq_b,freq_g]

def MeanCorrelation_old(Data,ws,N,wh = []):
    if len(wh)==0: wh = range(len(Data))
    abundsa = []
    means = []
    steps = np.arange(ws,ws+25*N,25)
    sers = Data[:,ws-ws:ws+ws]
    freq_red = rfftfreq(len(sers[0]),d=1e-3)
    for step in tqdm(steps,file=sys.stdout):
        ffts = []
        sers = Data[wh,step-ws:step+ws]
        abunds = []
        for i in range(len(sers)):
            FFT = np.abs(rfft(sers[i]))/len(sers[i])*2#,2)
            FFT = FFT[~np.isnan(FFT)]/np.sum(FFT)
            FFTn,FRQn = RemovePowerlaw2(FFT,freq_red)
            ffts.append(FFTn)
            abunds.append(Abundances(FFTn,FRQn))
        abundsa.append(np.nanmean(abunds,axis=0))
        means.append(np.nanmean(np.corrcoef(ffts)))
    return np.array(means),np.array(abundsa),np.array(steps)

def ProcessPower(RelativePower):
    Power_proc = []
    for i in range(len(RelativePower[0])):
        Sery = np.copy(RelativePower[:,i])
        Sery = Sery - np.nanmean(Sery)
        Sery = Sery / np.nanstd(Sery)
        Power_proc.append(Sery)
    return np.array(Power_proc)