# --------------------------------------------------------------------- #
# Preambule
# --------------------------------------------------------------------- #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# --------------------------------------------------------------------- #
print('Read')
# --------------------------------------------------------------------- #

Series = pd.read_pickle('/Users/mmdekker/Documents/Werk/Scripts/__Other/BrainProject/Mike/ProcData/Series.pkl')
ffts = pd.read_pickle('/Users/mmdekker/Documents/Werk/Scripts/__Other/BrainProject/Mike/ProcData/FFTs.pkl')
ffts_wm = pd.read_pickle('/Users/mmdekker/Documents/Werk/Scripts/__Other/BrainProject/Mike/ProcData/FFTs_wm.pkl')
Resids = pd.read_pickle('/Users/mmdekker/Documents/Werk/Scripts/__Other/BrainProject/Mike/ProcData/Resids.pkl')
NClusters = 7

# --------------------------------------------------------------------- #
print('Clustering (absolutesignal-) correlation matrix')
# --------------------------------------------------------------------- #

CorMats = np.corrcoef(Series)
KM = KMeans(n_clusters=NClusters, random_state=0).fit(CorMats)
Clusters_s = KM.labels_

# --------------------------------------------------------------------- #
print('Sorting (powerspectrum-residual-) correlation matrix')
# --------------------------------------------------------------------- #

CorMatf = np.corrcoef(Resids)
KM = KMeans(n_clusters=NClusters, random_state=0).fit(CorMatf)
Clusters_f = KM.labels_

# --------------------------------------------------------------------- #
print('Correspondence clusters')
# --------------------------------------------------------------------- #

corrcoef = []
dummycluss = []
for i in np.unique(Clusters_s):
    for j in np.unique(Clusters_s):
        dummyclus = np.copy(Clusters_s)
        dummyclus[dummyclus==i]=j*10
        dummyclus[dummyclus==j]=i
        dummyclus[dummyclus==j*10]=j
        corrcoef.append(np.corrcoef(dummyclus,Clusters_f)[0,1])
        dummycluss.append(dummyclus)

Clusters_sn = np.array(dummycluss)[np.array(corrcoef)==np.nanmax(corrcoef)][0]

# --------------------------------------------------------------------- #
print('Sorting')
# --------------------------------------------------------------------- #

Series['KM'] = Clusters_sn
Data = Series.sort_values(by=['KM'])
Data = Data.drop(['KM'],axis=1)
Data = np.array(Data)
CorMats2 = np.corrcoef(Data)

Resids['KM'] = Clusters_f
Data = Resids.sort_values(by=['KM'])
Data = Data.drop(['KM'],axis=1)
Data = np.array(Data)
CorMatf2 = np.corrcoef(Data)

# --------------------------------------------------------------------- #
print('Plotting')
# --------------------------------------------------------------------- #

fig = plt.figure(figsize=(7,6))
plt.pcolormesh(CorMats2,cmap = plt.cm.coolwarm,vmin=-1,vmax=1)
plt.xlim([0,len(CorMats)])
plt.ylim([0,len(CorMats)])
lens = []
for i in np.unique(Clusters_sn):
    lens.append(list(Clusters_sn).count(i))
    leni = sum(lens)
    plt.plot([leni,leni],[-1e3,1e3],'k',lw=1.5)
    plt.plot([-1e3,1e3],[leni,leni],'k',lw=1.5)
plt.colorbar()
plt.title('Correlation matrix of signals')
plt.show()

fig = plt.figure(figsize=(7,6))
plt.pcolormesh(CorMatf2,cmap = plt.cm.coolwarm,vmin=-1,vmax=1)
plt.xlim([0,len(CorMatf)])
plt.ylim([0,len(CorMatf)])
lens = []
for i in np.unique(Clusters_f):
    lens.append(list(Clusters_f).count(i))
    leni = sum(lens)
    plt.plot([leni,leni],[-1e3,1e3],'k',lw=1.5)
    plt.plot([-1e3,1e3],[leni,leni],'k',lw=1.5)
plt.colorbar()
plt.title('Correlation matrix of power spectra residuals')
plt.show()

print('Correlation of clusters:',np.round(np.corrcoef(Clusters_sn,Clusters_f)[0,1],2))