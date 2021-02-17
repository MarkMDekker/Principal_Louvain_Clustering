# --------------------------------------------------------------------- #
# Preambule
# --------------------------------------------------------------------- #

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os
os.chdir('/Users/mmdekker/Documents/Werk/Scripts/__General/General_Code')
from GeneralFunctions import LouvainClustering
os.chdir('/Users/mmdekker/Documents/Werk/Scripts/__General/Classes/MODS')
from Pygraphviz import *

# --------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------- #

DATE = '20200113'
PATH_PROCDATA = '/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/'
PATH_SAVE = ('/Users/mmdekker/Documents/Werk/Figures/Figures_Side/Brain/'
             'Pics_'+DATE+'/PDFs/')

# --------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------- #

ex = 'HR_HIP'
save = 1
tau = 30
res = 100
SAVE_STR = ex

# --------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------- #

PCs = np.array(pd.read_pickle(PATH_PROCDATA+'PCs/'+ex+'.pkl'))
Xpc = PCs[0]*np.sign(np.mean(PCs[0]))
Ypc = PCs[1]*np.sign(np.mean(PCs[1]))

# --------------------------------------------------------------------- #
# Retrieve phase-space, calculate locations and resulting PDFs
# --------------------------------------------------------------------- #

space = np.linspace(np.floor(np.min([Xpc, Ypc])),
                    np.floor(np.max([Xpc, Ypc]))+1, res)

H, xedges, yedges = np.histogram2d(Xpc, Ypc, bins=[space, space])
xedges2 = (xedges[:-1] + xedges[1:])/2.
yedges2 = (yedges[:-1] + yedges[1:])/2.
Xvec = xedges2
Yvec = yedges2
res2 = len(xedges2)

dicti = {}
dicti['Indices'] = range(res2*res2)
dicti['xi'] = res2*list(range(res2))
listje = []
for i in range(res2):
    listje = listje + res2*[i]
dicti['yi'] = listje
xco = []
yco = []
zco = []
for i in range(len(listje)):
    xco.append(xedges2[dicti['xi'][i]])
    yco.append(yedges2[dicti['yi'][i]])
    zco.append(H.T.flatten()[i])
dicti['xcoord'] = xco
dicti['ycoord'] = yco
dicti['zcoord'] = zco

DF = pd.DataFrame(dicti)

T1 = 0
T2 = len(Xpc)-tau

Rates = np.zeros(shape=(res2**2, res2**2))
DFindex = np.array(DF.index)
DFxi = np.array(DF.xi)
DFyi = np.array(DF.yi)
xe = np.array([xedges2]*len(Ypc[T1:T2]))
ye = np.array([yedges2]*len(Xpc[T1:T2]))

Locations = []
for j in [0, tau]:
    xe_arg = np.abs(xe.T-Ypc[T1+j:T2+j]).argmin(axis=0)
    ye_arg = np.abs(ye.T-Xpc[T1+j:T2+j]).argmin(axis=0)
    Locations.append(ye_arg*len(xedges2)+xe_arg)
Locations = np.array(Locations)

for i in tqdm(range(len(Locations[0]))):
    Rates[Locations[0, i], Locations[1, i]] += 1

Rates_n = []
for i in range(len(Rates)):
    Rates_n.append(np.copy(Rates[i])/np.nansum(Rates[i]))

Rates_n = np.array(np.nan_to_num(Rates_n, 0))

# --------------------------------------------------------------------- #
# Create and perform clustering (Louvain)
# --------------------------------------------------------------------- #


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


clusters, mod = LouvainClustering(Rates)
res2 = len(Xvec)

# --------------------------------------------------------------------- #
# Throw away smallest clusters
# --------------------------------------------------------------------- #

Clusters = []
for i in range(len(clusters)):
    if len(clusters[i]) > 15:
        Clusters.append(clusters[i])

Existence = []
for i in range(len(Clusters)):
    vec = func21(np.zeros(shape=(res2, res2)))
    vec[Clusters[i]] = 1
    Existence.append(vec)

Clustering = np.zeros(len(Existence[0]))
for i in range(len(Existence)):
    Clustering[Existence[i] == 1] = i+1
Clustering = func12(Clustering)

#%%
# --------------------------------------------------------------------- #
# Plot PDF onto phase-space
# --------------------------------------------------------------------- #

fig,ax = plt.subplots(figsize=(11,11))
for i in range(len(Existence)):
    ax.contour(xedges2, yedges2, func12(Existence[i]).T, range(-10,10), colors='k',
               linewidths=3, zorder=1e99)
ax.pcolormesh(xedges2, yedges2, H.T, zorder=-1, cmap=plt.cm.magma_r)

a=0
for i in range(len(Existence)):
    Zone = np.where(Existence[i]==1)[0]
    ax.text(Xvec[int(np.median(Zone/(len(space-1))))],Yvec[int(np.median(np.mod(Zone,len(space)-1)))],
                 str(a+1), fontsize=25, zorder=15, weight='bold', color='k',
                 ha='center', va='center',
                 bbox=dict(facecolor='w', alpha=0.2, edgecolor='k',lw=5,
                           boxstyle='round'))
    a+=1
ax.tick_params(axis='both', which='major', labelsize=12)
ax.tick_params(axis='both', which='minor', labelsize=12)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.tick_params(axis='both', which='minor', labelsize=14)
ax.set_ylim([-7, 7])
ax.set_xlim([-7, 7])
fig.tight_layout()
if save == 1:
    plt.savefig(PATH_SAVE+SAVE_STR+'.png', bbox_inches='tight', dpi=200)
