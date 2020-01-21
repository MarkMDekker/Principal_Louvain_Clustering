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

# --------------------------------------------------------------------- #
# ---------- Read
# --------------------------------------------------------------------- #

if not 'mat' in locals(): mat = scipy.io.loadmat('/Users/mmdekker/Documents/Werk/Data/Braindata/parLFP.mat')
Data = mat['EEG'][0][0]
Time = Data[14].T
Extra = Data[-1]
Series = np.array(Data[15])

# --------------------------------------------------------------------- #
# ---------- Some processing
# --------------------------------------------------------------------- #
Series2 = Series.T - Series.mean(axis=1)
stds = Series2.std(axis=0)
stds[stds==0] = 1
Series2 = Series2/stds

# --------------------------------------------------------------------- #
# ---------- Fourier Transformation
# --------------------------------------------------------------------- #

#fig,ax = plt.subplots(figsize=(8,5))
#
#ffts = []
#ffts_wm = []
#for i in tqdm(range(len(Series)),file=sys.stdout):
#    FFT = np.abs(rfft(Series[i]))/len(Series[i])*2
#    freq = rfftfreq(len(FFT),d=1)
#    #ax.plot(freq[:2000],windowmean(FFT[:2000],50))
#    ffts.append(FFT)
#    ffts_wm.append(windowmean(FFT,50))
#plt.show()
#
#pd.DataFrame(ffts).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/FFTs.pkl')
#pd.DataFrame(ffts_wm).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/FFTs_wm.pkl')

ffts = pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/FFTs.pkl')
ffts_wm = pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/FFTs_wm.pkl')

ffts_wm = np.array(ffts_wm)[:,25:]
ffts_wm = np.array(ffts_wm)[:,:-25]
freq = rfftfreq(len(ffts_wm[0]),d=1e-3)
ffts_wm = (ffts_wm.T/np.sum(ffts_wm,axis=1)).T


#%%
# --------------------------------------------------------------------- #
# QA - Preprocess data (remove 1/f fits)
# --------------------------------------------------------------------- #

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

Hz1 = np.where(freq>1)[0][0] # only from 1 Hz and larger
Hz2 = np.where(freq<=200)[0][-1] # only from 200 Hz and lower
frq = freq[Hz1:Hz2]
resids = []
for i in tqdm(range(len(ffts_wm)),file=sys.stdout):
    a,b = RemovePowerlaw(ffts_wm[i],freq)
    resids.append(a)
    frq = b
#
#pd.DataFrame(resids).to_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/Resids.pkl')

Resids = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/Resids.pkl'))


#%%
fig,ax = plt.subplots(figsize=(7,6))
for i in tqdm(range(len(Resids)),file=sys.stdout):
    ax.plot(frq,Resids[i])
ax.set_title('Power spectra')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Power')
ax.set_xlim([0,50])
ax.set_ylim([ax.get_ylim()[0],ax.get_ylim()[1]])
ax.plot([3,3],[-1e3,1e3],'k',lw=1.5)
ax.plot([8,8],[-1e3,1e3],'k',lw=1.5)
ax.plot([12,12],[-1e3,1e3],'k',lw=1.5)
ax.plot([38,38],[-1e3,1e3],'k',lw=1.5)
ax.plot([42,42],[-1e3,1e3],'k',lw=1.5)

ax.text(5.5,-1e-5,r'$\theta$',ha='center')
ax.text(10,-1e-5,r'$\alpha$',ha='center')
ax.text(25,-1e-5,r'$\beta$',ha='center')
ax.text(40,-1e-5,r'$\gamma$',ha='center')
plt.show()

#%%
# --------------------------------------------------------------------- #
# Q0 - Covariance
# --------------------------------------------------------------------- #

# Overall - covariance matrix
CovMat = Series2.T.dot(Series2)
fig = plt.figure(figsize=(7,6))
plt.pcolormesh(CovMat)
plt.colorbar()
plt.title('Covariance matrix - overall')
plt.show()

kmeans = KMeans(n_clusters=7, random_state=0).fit(CovMat)
Data = pd.DataFrame(np.array(Series2).T)
Data['KM'] = kmeans.labels_
Data = Data.sort_values(by=['KM'])
Data = Data.drop(['KM'],axis=1)
Data = np.array(Data).T
CovMat2 = Data.T.dot(Data)

fig = plt.figure(figsize=(7,6))
plt.pcolormesh(CovMat2)
plt.xlim([0,len(CovMat)])
plt.ylim([0,len(CovMat)])
lens = []
for i in np.unique(kmeans.labels_):
    lens.append(list(kmeans.labels_).count(i))
    leni = sum(lens)
    plt.plot([leni,leni],[-1e3,1e3],'k',lw=1.5)
    plt.plot([-1e3,1e3],[leni,leni],'k',lw=1.5)
plt.colorbar()
plt.title('Covariance matrix - overall (sorted)')
plt.show()

#%%
# --------------------------------------------------------------------- #
# Q1 - Frequency domain
# --------------------------------------------------------------------- #

# Correlations between frequency spectra

fig = plt.figure(figsize=(7,6))
plt.pcolormesh(freq[:10000],range(55),np.array(ffts_wm)[:,:10000])
plt.title('Frequency spectra, unsorted')
plt.show()

correlationmatrix = np.corrcoef(ffts_wm)
fig = plt.figure(figsize=(7,6))
c=plt.pcolormesh(correlationmatrix,vmax=1,cmap=plt.cm.rainbow)
plt.colorbar(c)
plt.title('Correlation matrix of frequency domains')
plt.show()

kmeans = KMeans(n_clusters=4, random_state=0).fit(correlationmatrix)
ffts2 = pd.DataFrame(np.array(ffts_wm))
ffts2['KM'] = kmeans.labels_
ffts2 = ffts2.sort_values(by=['KM'])
ffts2 = ffts2.drop(['KM'],axis=1)
correlationmatrix = np.corrcoef(ffts2)

fig = plt.figure(figsize=(7,6))
c=plt.pcolormesh(correlationmatrix,vmax=1,cmap=plt.cm.rainbow)
plt.xlim([0,len(correlationmatrix)])
plt.ylim([0,len(correlationmatrix)])
lens = []
for i in np.unique(kmeans.labels_):
    lens.append(list(kmeans.labels_).count(i))
    leni = sum(lens)
    plt.plot([leni,leni],[-1e3,1e3],'k',lw=1.5)
    plt.plot([-1e3,1e3],[leni,leni],'k',lw=1.5)
plt.colorbar(c)
plt.title('Sorted correlation matrix of frequency domains')
plt.show()

# Dominant frequency
Domfreq = []
for i in range(len(Series)):
    domfreq = freq[np.where(ffts_wm[i] == np.max(ffts_wm[i]))[0][0]]
    Domfreq.append(domfreq)
    
# Weighted frequency
Weightfreq = []
for i in range(len(Series)):
    weightfreq = freq.dot(ffts_wm[i])
    Weightfreq.append(weightfreq)
    
fig = plt.figure(figsize=(7,6))
plt.scatter(Domfreq,Weightfreq,c = plt.cm.rainbow(kmeans.labels_/np.max(kmeans.labels_)))
plt.ylabel('Weighted Frequency')
plt.xlabel('Dominant Frequency')
#plt.xlim([0.000005,0.0005])
plt.show()

#%%
# --------------------------------------------------------------------- #
# Q2 - Synchronization over time
# --------------------------------------------------------------------- #

def RemovePowerlaw(Data,freq):
    def fc(x, a, b):
        return x**a * b
    Hz1 = np.where(freq>=0.5)[0][0] # only from 1 Hz and larger
    Hz2 = np.where(freq<=200.5)[0][-1] # only from 200 Hz and lower
    frq = freq[Hz1:Hz2]
    data = Data[Hz1:Hz2]
    try:
        popt, pcov = curve_fit(fc, frq, data)
    except:
        popt = [np.nan,np.nan]
    residual = data - fc(frq,popt[0],popt[1])
    return np.array(residual),np.array(frq)

def Abundances(Data,frq):
    borders = np.arange(1,102,1)-0.5
    abund = []
    for i in range(len(borders)-1):
        abund.append(np.nanmean(Data[(frq>borders[i]) & (frq<=borders[i+1])]))
    return abund
        
def MeanCorrelation(Data,ws,N):
    abundsa = []
    means = []
    steps = np.arange(ws,ws+50*N,50)
    sers = Data[:,ws-ws:ws+ws]
    freq_red = rfftfreq(len(sers[0]),d=1e-3)
    for step in tqdm(steps,file=sys.stdout):
        ffts = []
        sers = Data[:,step-ws:step+ws]
        abunds = []
        for i in range(len(sers)):
            FFT = np.abs(rfft(sers[i]))/len(sers[i])*2#,2)
            FFT = FFT[~np.isnan(FFT)]/np.sum(FFT)
            FFTn,FRQn = RemovePowerlaw(FFT,freq_red)
            ffts.append(FFTn)
            abunds.append(Abundances(FFTn,FRQn))
        abundsa.append(np.nanmean(abunds,axis=0))
        means.append(np.nanmean(np.corrcoef(ffts)))
    return np.array(means),np.array(abundsa),np.array(steps)

Synchronization, RelativePower, Steps = MeanCorrelation(Series,500,10)

#%%
# Process frequency bands
Power_proc = []
for i in range(len(RelativePower[0])):
    Sery = np.copy(RelativePower[:,i])
    Sery = Sery - np.nanmean(Sery)
    Sery = Sery / np.nanstd(Sery)
    Power_proc.append(Sery)
Power_proc = np.array(Power_proc)
#%%

fig,(ax1,ax2) = plt.subplots(2,1,figsize=(12,12),sharex=True)
ax1.plot(Steps/1000.,Synchronization,'k',lw=2)
ax1.set_ylabel('Average of correlation matrix')
c=ax2.pcolormesh(Steps/1000.,range(100),Power_proc,vmin=-1,vmax=1,
               cmap = plt.cm.coolwarm)
#c=ax2.pcolormesh(Steps/1000.,range(100),Power_proc,vmin=0,vmax=0.002,
#               cmap = plt.cm.rainbow)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Frequency (Hz)')


plt.colorbar(c,orientation='horizontal',label='Normalized relative power')
fig.tight_layout

#%%
# --------------------------------------------------------------------- #
# Q3 - Retrieval of states
# --------------------------------------------------------------------- #

# Look at the four abundances -> patterns?
Data = np.copy(Power_proc)
#for i in range(len(Data)):
#    Data[i] = Data[i] - np.mean(Data[i])
#    Data[i] = Data[i]/np.std(Data[i])

vals,vecs = np.linalg.eig(Data.dot(Data.T))
idx = vals.argsort()[::-1]
vals_num = vals/np.sum(vals)
vals = vals[idx]
vecs = vecs[:,idx]
EOFs = []
PCs = []
for i in range(len(Data)):
    EOFs.append(vecs[:,i])
    PCs.append(vecs[:,i].dot(Data))

# Rescale Synchronization
Mvec = np.copy(Synchronization)
Mvec = Mvec - np.min(Mvec)
Mvec = Mvec / np.max(Mvec)

WM = 5
x = 0
y = 1
fig,ax = plt.subplots(figsize=(8,8))
ax.plot(windowmean(PCs[x],WM),windowmean(PCs[y],WM),'k',lw=0.2)
c=ax.scatter(windowmean(PCs[x],WM),windowmean(PCs[y],WM),c=plt.cm.rainbow(Mvec),zorder=5,s=25)
ax.set_xlabel('Principal Component 1 ('+str(int(vals_num[x]*100))+'%)')
ax.set_ylabel('Principal Component 2 ('+str(int(vals_num[y]*100))+'%)')
axc     = fig.add_axes([1., 0.1, 0.025, 0.8])
norm    = mpl.colors.Normalize(vmin=np.min(Synchronization), vmax=np.max(Synchronization))
cb1     = mpl.colorbar.ColorbarBase(axc, cmap=plt.cm.rainbow,norm=norm,extend='both',orientation='vertical')
cb1.set_label('Correlation matrix average',fontsize=15)
axc.tick_params(labelsize=13)
fig.tight_layout()

#%%
# --------------------------------------------------------------------- #
# Q4 - Multivariate regression
# --------------------------------------------------------------------- #

from sklearn import linear_model
import statsmodels.api as sm

X = np.array([abunds_a[:,0],abunds_a[:,1],abunds_a[:,2],abunds_a[:,3]]).T
Y = means
bord = 3800
X_train = X[:bord]
Y_train = Y[:bord]
X_test = X[:]
Y_test = Y[:]

# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# prediction with sklearn
print ('Predicted Synchronization: \n', regr.predict([[0.12,0.05,0.1,0.05]]))

# with statsmodels
X_train = sm.add_constant(X_train) # adding a constant
X_test = sm.add_constant(X_test) # adding a constant
 
model = sm.OLS(Y_train, X_train).fit()
predictions = model.predict(X_test) 
 
print_model = model.summary()
print(print_model)

# prediction with sklearn
print ('Predicted Synchronization: \n', regr.predict([[0.12,0.05,0.1,0.05]]))

fig,ax = plt.subplots(figsize=(15,6))
ax.plot(Y_test,label='Observed')
ax.plot(predictions,label='Predicted')
ax.legend()