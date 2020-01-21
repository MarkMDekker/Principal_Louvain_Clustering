# =========================================================================== #
# Functions for the brain project
# =========================================================================== #

import numpy as np
#import time
from tqdm import tqdm
import sys
from scipy.fftpack import rfftfreq, rfft
from scipy.optimize import curve_fit

# =========================================================================== #
# Core function
# =========================================================================== #

def MeanCorrelation(Data,ws,N,wh = []):
    
    if len(wh)==0: wh = range(len(Data))
    #T0 = time.time()
    abundsa = []
    means = []
    steps = np.arange(ws,N-ws)
    sers = Data[:,ws-ws:ws+ws]
    freq_red = rfftfreq(len(sers[0]),d=1e-3)
    
    Hz1 = np.where(freq_red>=0.5)[0][0] # only from 1 Hz and larger
    Hz2 = np.where(freq_red<=100.5)[0][-1] # only from 200 Hz and lower
    frq = freq_red[Hz1:Hz2]
    
    borders = np.arange(0,101,2)
    wheres = []
    ran = np.arange(len(frq))
    for i in range(len(borders)-1):
        wheres.append(ran[(frq>borders[i]) & (frq<=borders[i+1])])
    
    #T1 = time.time() - T0
    #T3 = []
    #T4 = []
    #T5 = []
    #T6 = []
    #print('T1',T1)
    for step in tqdm(steps,file=sys.stdout):
        ffts = []
        sers = Data[wh,step-ws:step+ws]
        abunds = []
        for i in range(len(sers)):
            #T2 = time.time() - T0
            FFT = np.abs(rfft(sers[i]))/len(sers[i])*2#,2)
            FFT = FFT[~np.isnan(FFT)]/np.sum(FFT)
            #T3a = time.time() - T0
            #T3.append(T3a-T2)
            FFTn = RemovePowerlaw2(FFT,frq,Hz1,Hz2)
            #T4a = time.time() - T0
            #T4.append(T4a - T3a)
            ffts.append(FFTn)
            abunds.append(Abundances(FFTn,frq,borders,wheres))
            #T5a = time.time() - T0
            #T5.append(T5a - T4a)
        abundsa.append(np.nanmean(abunds,axis=0))
        means.append(np.nanmean(np.corrcoef(ffts)))
        #T6a = time.time() - T0
        #T6.append(T6a - T5a)
        
    #print('T3',np.mean(T3))
    #print('T4',np.mean(T4))
    #print('T5',np.mean(T5))
    #print('T6',np.mean(T6))
    return np.array(means),np.array(abundsa),np.array(steps)
    
# =========================================================================== #
# Helper functions
# =========================================================================== #

def Abundances(Data,frq,borders,wheres):
    abund = []
    for i in range(len(borders)-1):
        abund.append(np.mean(Data[wheres[i]]))
    return abund

def fc(x, a, b):
    return x**a * b
    
def RemovePowerlaw2(Data,frq,Hz1,Hz2):
    data = Data[Hz1:Hz2]
    try:
        popt, pcov = curve_fit(fc, frq, data,p0 = [-0.5, 0.02])
    except:
        popt = [np.nan,np.nan]
    ind = 0
    while len(frq)!=len(data) and ind<10:
        data = np.array(list(data)+[0])
        ind +=1
    residual = data - fc(frq,popt[0],popt[1])
    return np.array(residual)

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

def RemovePowerlaw(Data,freq):
    def fc(x, a, b):
        return x**a * b
    Hz1 = np.where(freq>1)[0][0] # only from 1 Hz and larger
    Hz2 = np.where(freq<=100)[0][-1] # only from 200 Hz and lower
    frq = freq[Hz1:Hz2]
    data = Data[Hz1:Hz2]                            # !
    popt, pcov = curve_fit(fc, frq, data)
    residual = data - fc(frq,popt[0],popt[1])
    return np.array(residual),np.array(frq)         # !

def Standardize(Data):
    Data = Data.T
    Data = Data.T - Data.mean(axis=1)
    stds = Data.std(axis=0)
    stds[stds==0] = 1
    Data = Data/stds
    return Data

def FourierTrans(Data,string):
    ffts = []
    ffts_wm = []
    for i in tqdm(range(len(Data)),file=sys.stdout):
        FFT = np.abs(rfft(Data[i]))/len(Data[i])*2
        freq = rfftfreq(len(FFT),d=1)
        ffts.append(FFT)
        ffts_wm.append(windowmean(FFT,50))
    pd.DataFrame(ffts).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/FFTs_'+string+'.pkl')
    pd.DataFrame(ffts_wm).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/FFTs_wm_'+string+'.pkl')

def Powerlaw(ffts_wm,string,freq):
    Hz1 = np.where(freq>1)[0][0] # only from 1 Hz and larger
    Hz2 = np.where(freq<=200)[0][-1] # only from 200 Hz and lower
    frq = freq[Hz1:Hz2]
    resids = []
    for i in tqdm(range(len(ffts_wm)),file=sys.stdout):
        a,b = RemovePowerlaw(ffts_wm[i],freq)
        resids.append(a)
        frq = b
    pd.DataFrame(resids).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/Resids_'+string+'.pkl')
    pd.DataFrame(frq).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/Resids_frq.pkl')
        
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