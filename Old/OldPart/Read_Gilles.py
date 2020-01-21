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
fig,ax = plt.subplots(figsize=(7,6))
for i in range(len(ffts_wm)):
    ax.plot(freq,ffts_wm[i])
ax.set_title('Power spectra')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Power')
ax.set_xlim([0,50])
ax.set_ylim([ax.get_ylim()[0],ax.get_ylim()[1]])
ax.plot([3,3],[-1e3,1e3],'k',lw=0.5)
ax.plot([8,8],[-1e3,1e3],'k',lw=0.5)
ax.plot([12,12],[-1e3,1e3],'k',lw=0.5)
ax.plot([38,38],[-1e3,1e3],'k',lw=0.5)
ax.plot([42,42],[-1e3,1e3],'k',lw=0.5)

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

kmeans = KMeans(n_clusters=4, random_state=0).fit(CovMat)
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

ffts0 = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/FFTs.pkl'))

#Only within label 0
Series_lab0 = Series[kmeans.labels_==0]
Series_lab1 = Series[kmeans.labels_==1]
Series_lab2 = Series[kmeans.labels_==2]
Series_lab3 = Series[kmeans.labels_==3]
corrmat = []
means=[]
corrmat0 = []
means0=[]
corrmat1 = []
means1=[]
corrmat2 = []
means2=[]
corrmat3 = []
means3=[]

abunds_a = []
abunds_0 = []
abunds_1 = []
abunds_2 = []
abunds_3 = []

steps = np.arange(1500,1500+100*400,100)
sers = Series[:,1500-1500:1500+1500]
freq_red = rfftfreq(len(sers[0]),d=1e-3)
for step in tqdm(steps,file=sys.stdout):
    ffts = []
    sers = Series[:,step-1500:step+1500]
    abunds = []
    for i in range(len(sers)):
        FFT = np.abs(rfft(sers[i]))/len(sers[i])*2#,2)
        FFT = FFT[~np.isnan(FFT)]/np.sum(FFT)
        ffts.append(FFT)
        abunds.append(relabund(FFT,freq_red))
    abunds_a.append(np.mean(abunds,axis=0))
    corrmat.append(np.corrcoef(ffts))
    means.append(np.mean(corrmat[-1]))
    
    ffts = []
    sers = Series_lab0[:,step-1500:step+1500]
    for i in range(len(sers)):
        FFT = np.abs(rfft(sers[i]))/len(sers[i])*2#,2)
        FFT = FFT[~np.isnan(FFT)]/np.sum(FFT)
        ffts.append(FFT)
        abunds.append(relabund(FFT,freq_red))
    abunds_0.append(np.mean(abunds,axis=0))
    corrmat0.append(np.corrcoef(ffts))
    means0.append(np.mean(corrmat0[-1]))
    
    ffts = []
    sers = Series_lab1[:,step-1500:step+1500]
    for i in range(len(sers)):
        FFT = np.abs(rfft(sers[i]))/len(sers[i])*2#,2)
        FFT = FFT[~np.isnan(FFT)]/np.sum(FFT)
        ffts.append(FFT)
        abunds.append(relabund(FFT,freq_red))
    abunds_1.append(np.mean(abunds,axis=0))
    corrmat1.append(np.corrcoef(ffts))
    means1.append(np.mean(corrmat1[-1]))
    
    ffts = []
    sers = Series_lab2[:,step-1500:step+1500]
    for i in range(len(sers)):
        FFT = np.abs(rfft(sers[i]))/len(sers[i])*2#,2)
        FFT = FFT[~np.isnan(FFT)]/np.sum(FFT)
        ffts.append(FFT)
        abunds.append(relabund(FFT,freq_red))
    abunds_2.append(np.mean(abunds,axis=0))
    corrmat2.append(np.corrcoef(ffts))
    means2.append(np.mean(corrmat2[-1]))
    
    ffts = []
    sers = Series_lab3[:,step-1500:step+1500]
    for i in range(len(sers)):
        FFT = np.abs(rfft(sers[i]))/len(sers[i])*2#,2)
        FFT = FFT[~np.isnan(FFT)]/np.sum(FFT)
        ffts.append(FFT)
        abunds.append(relabund(FFT,freq_red))
    abunds_3.append(np.mean(abunds,axis=0))
    corrmat3.append(np.corrcoef(ffts))
    means3.append(np.mean(corrmat3[-1]))

abunds_a = np.array(abunds_a)
abunds_0 = np.array(abunds_0)
abunds_1 = np.array(abunds_1)
abunds_2 = np.array(abunds_2)
abunds_3 = np.array(abunds_3)

#%%
fig,(ax1,ax2) = plt.subplots(2,1,figsize=(12,6),sharex=True)
ax1.plot(steps/1000.,means0,label='Cluster 0',lw=0.5)
ax1.plot(steps/1000.,means1,label='Cluster 1',lw=0.5)
ax1.plot(steps/1000.,means2,label='Cluster 2',lw=0.5)
ax1.plot(steps/1000.,means3,label='Cluster 3',lw=0.5)
ax1.plot(steps/1000.,means,'k',label='Overall',lw=2)
ax1.legend()
ax2.set_xlabel('Time (s)')
ax1.set_ylabel('Average of correlation matrix')
ax2.plot(steps/1000.,abunds_0[:,0],label='Cluster 0',lw=0.5)
ax2.plot(steps/1000.,abunds_1[:,0],label='Cluster 1',lw=0.5)
ax2.plot(steps/1000.,abunds_2[:,0],label='Cluster 2',lw=0.5)
ax2.plot(steps/1000.,abunds_3[:,0],label='Cluster 3',lw=0.5)
ax2.plot(steps/1000.,abunds_a[:,0],'k',label='Overall',lw=2)
ax2.set_ylabel('Relative power in theta spectrum')

#%%
# --------------------------------------------------------------------- #
# Q3 - Retrieval of states
# --------------------------------------------------------------------- #

# Look at the four abundances -> patterns?
Data = np.copy(abunds_0).transpose()
for i in range(4):
    Data[i] = Data[i] - np.mean(Data[i])
    Data[i] = Data[i]/np.std(Data[i])

vals,vecs = np.linalg.eig(Data.dot(Data.T))
vals_num = vals/np.sum(vals)
EOF0 = vecs[:,0]
EOF1 = vecs[:,1]
EOF2 = vecs[:,3] # int 3 geeft meer %!
EOF3 = vecs[:,2]

PC0 = EOF0.dot(Data)
PC1 = EOF1.dot(Data)
PC2 = EOF2.dot(Data)
PC3 = EOF3.dot(Data)

#rescale means
Mvec = np.copy(means)
Mvec = Mvec - np.min(Mvec)
Mvec = Mvec / np.max(Mvec)

fig,ax = plt.subplots(figsize=(8,8))
ax.plot(PC0,PC1,'k',lw=0.2)
c=ax.scatter(PC0,PC1,c=plt.cm.rainbow(Mvec),zorder=5,s=25)
ax.set_xlabel('Principal Component 1 ('+str(int(vals_num[0]*100))+'%)')
ax.set_ylabel('Principal Component 2 ('+str(int(vals_num[1]*100))+'%)')
axc     = fig.add_axes([1., 0.1, 0.025, 0.8])
norm    = mpl.colors.Normalize(vmin=np.min(Mvec), vmax=np.max(Mvec))
cb1     = mpl.colorbar.ColorbarBase(axc, cmap=plt.cm.rainbow,norm=norm,extend='both',orientation='vertical')
cb1.set_label('Synchronization',fontsize=15)
axc.tick_params(labelsize=13)
fig.tight_layout()


#%%
from scipy.fftpack import rfftfreq, rfft
import numpy as np
import matplotlib.pyplot as plt


def spectrum(sig, t):
    f = rfftfreq(sig.size, d=t[1]-t[0])
    y = rfft(sig)
    return f, y


if __name__ == "__main__":
    N = 1000
    t = np.linspace(0, 4*np.pi, N)
    x = np.sin(10*np.pi*t)+np.random.randn(N)


    f, y = spectrum(x, t)

    plt.subplot(211)
    plt.plot(t,x)
    plt.grid(True)
    plt.subplot(212)
    y_a = np.abs(y)/N*2 # Scale accordingly
    plt.plot(f, y_a)
    plt.grid(True)
    plt.show()