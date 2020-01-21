# --------------------------------------------------------------------- #
# Preambule
# --------------------------------------------------------------------- #

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import networkx as nx
import seaborn as sns
import sys
from community import community_louvain
from community import modularity
exec(open('/Users/mmdekker/Documents/Werk/Scripts/Classes/MODS/Pygraphviz.py').read())

# --------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------- #

date    = '20200113'
ex      = 'HIP'
tau     = 30
res     = 25
save    = 1
string  = ex+'_tau'+str(int(tau))+'_res'+str(int(res))

# --------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------- #

PP              = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/ProcessedPower/'+ex+'_hr.pkl'))
EOFs            = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/EOFs/D_'+ex+'.pkl'))
PCs = []
for i in range(2):
    PCs.append(EOFs[i].dot(PP))
Xpc             = PCs[0]
Ypc             = PCs[1]

# --------------------------------------------------------------------- #
# Functions
# --------------------------------------------------------------------- #

def LouvainClustering(Ratesmat,red=0):   
    ConMat = Ratesmat+Ratesmat.T
    vec = np.array(np.where(np.sum(ConMat,axis=0)>0)[0])
    AdjMat = Ratesmat[vec][:,vec]
    inds = np.arange(len(AdjMat))
    #nodes = range(0,len(AdjMat))
    edges = []
    weights = []
    for i in range(0,len(AdjMat)):
        nonzeros = inds[(inds>=i+1) & (AdjMat[i,inds]>0) & (inds!=i)]
        for j in nonzeros:#range(i+1,len(AdjMat)):
            #if i!=j:# and np.mean([AdjMat[i,j],AdjMat[j,i]])>0:
            edges.append((i,j))
            weights.append(np.mean([AdjMat[i,j],AdjMat[j,i]]))
    
    Networkx = nx.Graph()
    Networkx.add_nodes_from(np.unique(edges))
    Networkx.add_edges_from(edges)
    partition = community_louvain.best_partition(Networkx,weight='weight')
    mod = modularity(partition,Networkx,weight='weight')
    
    if red==0:
        count = 0.
        clusters = []
        for com in set(partition.values()) :
            count = count + 1.
            list_nodes = [nodes for nodes in partition.keys()
                                        if partition[nodes] == com]
            clusters.append(vec[list_nodes])
        return clusters, mod
    else:
        return mod

def func12(vec):
    mat = np.zeros(shape=(res2,res2))
    a=0
    for i in range(res2):
        for j in range(res2):
            mat[i,j] = vec[a]
            a+=1
    return mat
    
def func21(mat):
    return mat.flatten()

# --------------------------------------------------------------------- #
# Initialize iterations over tau and res
# --------------------------------------------------------------------- #

Modularities = []
taus = np.arange(1,300,3)
ress = np.arange(10,100.1,5)

for tau in tqdm(taus):
    Mods = []
    for res in ress:
        
        # --------------------------------------------------------------------- #
        # Retrieve phase-space
        # --------------------------------------------------------------------- #
        
        space = np.linspace(np.floor(np.min([Xpc,Ypc])),np.floor(np.max([Xpc,Ypc]))+1,res)
        
        H, xedges, yedges = np.histogram2d(Xpc, Ypc, bins=[space,space])
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
            listje = listje+ res2*[i]
        dicti['yi'] = listje
        xco = []
        yco = []
        #zco = []
        for i in range(len(listje)):
            xco.append(xedges2[dicti['xi'][i]])
            yco.append(yedges2[dicti['yi'][i]])
            #zco.append(H.T.flatten()[i])
        dicti['xcoord'] = xco
        dicti['ycoord'] = yco
        #dicti['zcoord'] = zco
        
        DF = pd.DataFrame(dicti)
        
        # --------------------------------------------------------------------- #
        # Calculate locations
        # --------------------------------------------------------------------- #
        
        T1 = 0
        T2 = len(Xpc)-tau
        DFindex = np.array(DF.index)
        DFxi = np.array(DF.xi)
        DFyi = np.array(DF.yi)
        xe = np.array([xedges2]*len(Ypc[T1:T2]))
        ye = np.array([yedges2]*len(Xpc[T1:T2]))
        
        Locations = []
        for j in [0,tau]:
            xe_arg = np.abs(xe.T-Ypc[T1+j:T2+j]).argmin(axis=0)
            ye_arg = np.abs(ye.T-Xpc[T1+j:T2+j]).argmin(axis=0)
            Locations.append(ye_arg*len(xedges2)+xe_arg)
        Locations = np.array(Locations)
        
        # --------------------------------------------------------------------- #
        # Calculate rates
        # --------------------------------------------------------------------- #
        
        Rates = np.zeros(shape=(res2**2,res2**2))
        for i in range(len(Locations[0])):
            Rates[Locations[0,i],Locations[1,i]] +=1
        
        Rates_n = np.copy(Rates)/(1e-9+np.nansum(Rates,axis=1))
        #Rates_n = np.array(np.nan_to_num(Rates_n,0))
        
        # --------------------------------------------------------------------- #
        # Apply clustering algorithm
        # --------------------------------------------------------------------- #
        
        mod = LouvainClustering(Rates_n,1)
        Mods.append(mod)
    Modularities.append(Mods)

#%%
# --------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------- #

import scipy.ndimage as ndimage

Z = ndimage.gaussian_filter(np.array(Modularities), sigma=1, order=0)

fig, ax = plt.subplots(figsize=(8,8))
ax.contourf(taus,ress**2.,np.array(Z).T,25,cmap=plt.cm.magma_r)
ax.contour(taus,ress**2.,np.array(Z).T,25,colors='k')
ax.set_ylabel('Amount of gridcells',fontsize=15)
ax.set_xlabel(r'Transition time $\tau$',fontsize=15)
ax.set_xlim([ax.get_xlim()[0],ax.get_xlim()[1]])
ax.set_ylim([ax.get_ylim()[0],ax.get_ylim()[1]])

#for i in ress**2.:
#    ax.plot([-1e3,1e3],[i,i],'k:',lw=0.5,zorder=1e9)
#for i in taus:
#    ax.plot([i,i],[-1e3,1e5],'k:',lw=0.5,zorder=1e9)

axc     = fig.add_axes([1, 0.1, 0.015, 0.85])
norm    = mpl.colors.Normalize(vmin=np.min(Modularities), vmax=np.max(Modularities))
cb1     = mpl.colorbar.ColorbarBase(axc, cmap=plt.cm.magma_r,extend='max',norm=norm,orientation='vertical')
cb1.set_label(r'Modularity',fontsize=12)
axc.tick_params(labelsize=12)

fig.tight_layout()
if save==1: plt.savefig('/Users/mmdekker/Documents/Werk/Figures/Neurostuff/Pics_'+date+'/Sensitivities/Sens_'+ex+'.png',bbox_inches = 'tight',dpi=200)
plt.show()