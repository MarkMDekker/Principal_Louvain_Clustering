# --------------------------------------------------------------------- #
# Preambule
# --------------------------------------------------------------------- #

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import sys

# --------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------- #

save        = 1
date        = '20200113'
ex          = 'HR_PAR'
PCs         = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/PCs/'+ex+'.pkl'))
#PC1s        = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/PCs/Cutsets/HR_'+ex+'_pc1.pkl'))
#PC2s        = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/PCs/Cutsets/HR_'+ex+'_pc2.pkl'))
#if cho=='old': ind = 2
#if cho=='new': ind = 1
#if cho=='gen': ind = 0
#if cho=='notexpl': ind = 3

K = 0

Xpc = PCs[0]*np.sign(np.mean(PCs[0]))
Ypc = PCs[1]*np.sign(np.mean(PCs[1]))
tau = 30
res = 100
string=ex+'_cs'#ex+'_tau'+str(int(tau))+'_res'+str(int(res))
#%%
# --------------------------------------------------------------------- #
# Retrieve phase-space, calculate locations and rates
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

Rates = np.zeros(shape=(res2**2,res2**2))
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

for i in tqdm(range(len(Locations[0])),file=sys.stdout):#tqdm(range(len(pci)-np.max(tau)),file=sys.stdout):
    Rates[Locations[0,i],Locations[1,i]] +=1

Rates_n = []
for i in range(len(Rates)):
    Rates_n.append(np.copy(Rates[i])/np.nansum(Rates[i]))
        
Rates_n = np.array(np.nan_to_num(Rates_n,0))

#%%
# --------------------------------------------------------------------- #
# Create and perform clustering (Louvain)
# --------------------------------------------------------------------- #

exec(open('/Users/mmdekker/Documents/Werk/Scripts/Classes/MODS/Pygraphviz.py').read())
from community import community_louvain
from community import modularity

def LouvainClustering(Ratesmat):   
        
    ConMat = Ratesmat+Ratesmat.T
    vec = np.array(np.where(np.sum(ConMat,axis=0)>0)[0])
    AdjMat = Ratesmat[vec][:,vec]
    
    nodes = range(0,len(AdjMat))
    edges = []
    weights = []
    for i in tqdm(range(0,len(AdjMat)),file=sys.stdout):
        for j in range(i+1,len(AdjMat)):
            if i!=j and np.mean([AdjMat[i,j],AdjMat[j,i]])>0:
                edges.append((i,j))
                weights.append(np.mean([AdjMat[i,j],AdjMat[j,i]]))
    
    Networkx = nx.Graph()
    Networkx.add_nodes_from(np.unique(edges))
    Networkx.add_edges_from(edges)
    
    partition = community_louvain.best_partition(Networkx,weight='weight')
    mod = modularity(partition,Networkx,weight='weight')
    
    
    count = 0.
    clusters = []
    for com in set(partition.values()) :
        count = count + 1.
        list_nodes = [nodes for nodes in partition.keys()
                                    if partition[nodes] == com]
        clusters.append(vec[list_nodes])
    
    return clusters, mod

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

#LouvainClustering(Rates_n)

# --------------------------------------------------------------------- #
# Apply clustering algorithm
# --------------------------------------------------------------------- #

clusters, mod = LouvainClustering(Rates)
res2 = len(Xvec)
#pd.DataFrame(clusters).to_pickle(directory+'Clusters/Highres_'+str(TimeLag)+'.pkl')

# --------------------------------------------------------------------- #
# Throw away smallest clusters
# --------------------------------------------------------------------- #

Clusters = []
for i in range(len(clusters)):
    if len(clusters[i])>15:
        Clusters.append(clusters[i])
Existence = []
for i in range(len(Clusters)):
    vec = func21(np.zeros(shape=(res2,res2)))+np.nan
    vec[Clusters[i]] = 1
    Existence.append(vec)

#%%

# List number of new-exploration 
New2  = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/NewOld/New_hr.pkl')).T[0]
New2 = New2[:len(Locations[0])]
Locs_of_new = Locations[0]*New2
Locs_of_new = Locs_of_new[Locs_of_new>0].astype(int)
Num_new = np.zeros(len(Clusters))
for i in range(len(Locs_of_new)):
    try:
        Num_new[np.where(np.array(Existence)[:,Locs_of_new[i]]==1)[0][0]] += 1
    except:
        continue
    
# List number of old-exploration 
Old2  = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/NewOld/Old_hr.pkl')).T[0]
Old2 = Old2[:len(Locations[0])]
Locs_of_old = Locations[0]*Old2
Locs_of_old = Locs_of_old[Locs_of_old>0].astype(int)
Num_old = np.zeros(len(Clusters))
for i in range(len(Locs_of_old)):
    try:
        Num_old[np.where(np.array(Existence)[:,Locs_of_old[i]]==1)[0][0]] += 1
    except:
        continue
    
# List number of exploration in general
Exp2  = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/NewOld/Exp_hr.pkl')).T[0]
Exp2 = Exp2[:len(Locations[0])]
Locs_of_exp = Locations[0]*Exp2
Locs_of_exp = Locs_of_exp[Locs_of_exp>0].astype(int)
Num_exp = np.zeros(len(Clusters))
for i in range(len(Locs_of_exp)):
    try:
        Num_exp[np.where(np.array(Existence)[:,Locs_of_exp[i]]==1)[0][0]] += 1
    except:
        continue
    
# List number of all
Locs = Locations[0]
Locs = Locs[Locs>0].astype(int)
Num = np.zeros(len(Clusters))
for i in range(len(Locs)):
    try:
        Num[np.where(np.array(Existence)[:,Locs[i]]==1)[0][0]] += 1
    except:
        continue

Score_new = Num_new/Num/(np.nansum(Num_new)/np.nansum(Num))
Score_old = Num_old/Num/(np.nansum(Num_old)/np.nansum(Num))
Score_exp = Num_exp/Num/(np.nansum(Num_exp)/np.nansum(Num))
Score_num = Num/np.nansum(Num)

# Average synchronization
Synchronization = np.array(pd.read_pickle('/Users/mmdekker/Documents/Werk/Data/Braindata/ProcessedData/Synchronization/'+ex+'.pkl')).T[0]

# List number of all
Locs = Locations[0]
Locs = Locs[Locs>0].astype(int)
Syn_cl = np.zeros(len(Clusters))
for i in range(len(Locs)):
    try:
        Syn_cl[np.where(np.array(Existence)[:,Locs[i]]==1)[0][0]] += Synchronization[i]
    except:
        continue

Syn_cl = Syn_cl / Num

##%%
## Transition probabilities
#
#TransMat = np.zeros(shape=(len(Clusters),len(Clusters)))
#for i in range(len(Clusters)):
#    for j in range(len(Clusters)):
#        TransMat[i,j] = np.nansum(Rates_n[Clusters[i]][:,Clusters[j]])/np.nansum(Rates_n[Clusters[i]])
#
#fig,ax = plt.subplots(figsize=(10,8))
#c=ax.pcolormesh(TransMat,vmin=0,vmax=1,cmap=plt.cm.jet)
#bar = plt.colorbar(c)
#bar.set_label('Transition probability')
#ax.set_xlabel('From')
#ax.set_ylabel('To (5 sec later)')
#figdir = '/Users/mmdekker/Documents/Werk/Figures/Neurostuff/Dataset_new/'
##plt.savefig(figdir+'TransMat_'+string[-3:]+'.png',bbox_inches = 'tight',dpi=200)

#%%
# --------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------- #

colors = ['forestgreen','tomato','steelblue','gold','violet',
          'silver','gray','firebrick','chocolate','k',
          'k']
fig,ax = plt.subplots(figsize=(11,11))

# ====> All subclusters
a=0
subclusX = []
subclusY = []
for i in range(len(Existence)):
    palette = sns.light_palette(colors[i],as_cmap=True)
    #ax.pcolormesh(Xvec,Yvec,func12(Existence[i]).T,facecolor=colors[i])#,edgecolors='k',cmap='gray_r',vmin=-10,vmax=-9,lw=9)
    ax.pcolormesh(Xvec,Yvec,func12(Existence[i]).T,edgecolors='k',cmap=palette,vmin=-10,vmax=-9,lw=0)
#    for i in range(len(subexistences[toclus])):
    Zone = np.where(Existence[i]==1)[0]
#        ax.pcolormesh(Xvec,Yvec,func12(subexistences[toclus][i]).T,facecolor='none',edgecolors='k',cmap='gray_r',vmin=-10,vmax=-9,lw=5)
#        ticks = np.linspace(-1,1,16)
#        var = func12(subexistences[toclus][i]).T
#        palette = sns.light_palette(colors[toclus+1],as_cmap=True)
#        c = ax.pcolormesh(Xvec,Yvec,var,facecolor=colors[toclus+1],cmap=palette,vmin=0,vmax=0.9+0.4*i)#,vmin=-1,vmax=1, linewidth=0.0015625)#,norm = norm)
    ax.text(Xvec[int(np.median(Zone/(len(space-1))))],Yvec[int(np.median(np.mod(Zone,len(space)-1)))],
                 str(a+1),fontsize=15,zorder=15,weight='bold',color='white',
                 ha='center',va='center',
                 bbox=dict(facecolor='k',alpha=0.7, edgecolor=colors[i],lw=5, boxstyle='round'))
#        subclusX.append(Xvec[int(np.median(Zone/(len(Gridding)-1)))])
#        subclusY.append(Xvec[int(np.median(np.mod(Zone,len(Gridding)-1)))])
    a+=1

# List scores
ax.text(0.01,0.93+0.03,'Clus',transform=ax.transAxes)
ax.text(0.06,0.93+0.03,'Old',transform=ax.transAxes)
ax.text(0.11,0.93+0.03,'New',transform=ax.transAxes)
ax.text(0.16,0.93+0.03,'Exp',transform=ax.transAxes)
ax.text(0.21,0.93+0.03,'Perc',transform=ax.transAxes)
ax.text(0.26,0.93+0.03,'Synchr',transform=ax.transAxes)
for i in range(len(Clusters)):
    ax.text(0.01,0.93-0.03*i,str(i+1)+': ',transform=ax.transAxes)
    ax.text(0.06,0.93-0.03*i,str(np.round(Score_old[i],3)),transform=ax.transAxes)
    ax.text(0.11,0.93-0.03*i,str(np.round(Score_new[i],3)),transform=ax.transAxes)
    ax.text(0.16,0.93-0.03*i,str(np.round(Score_exp[i],3)),transform=ax.transAxes)
    ax.text(0.21,0.93-0.03*i,str(np.round(100*Score_num[i],1))+'%',transform=ax.transAxes)
    ax.text(0.26,0.93-0.03*i,str(np.round(Syn_cl[i],3)),transform=ax.transAxes)

ax.text(0.06,0.93-0.03*(i+1),str(np.round(100*np.nansum(Num_old)/np.nansum(Num),1))+'%',transform=ax.transAxes)
ax.text(0.11,0.93-0.03*(i+1),str(np.round(100*np.nansum(Num_new)/np.nansum(Num),1))+'%',transform=ax.transAxes)
ax.text(0.16,0.93-0.03*(i+1),str(np.round(100*np.nansum(Num_exp)/np.nansum(Num),1))+'%',transform=ax.transAxes)
ax.text(0.26,0.93-0.03*(i+1),str(np.round(np.mean(Synchronization),3)),transform=ax.transAxes)
ax.text(0.99,0.99,'Modularity: '+str(np.round(mod,3)),ha='right',va='top',transform=ax.transAxes,fontsize=15)

maxy = ax.get_ylim()[1]
miny = ax.get_ylim()[0]

ax.tick_params(axis='both', which='major', labelsize=12)
ax.tick_params(axis='both', which='minor', labelsize=12)
ax.set_xlabel('Principal Component 1',fontsize=15)
ax.set_ylabel('Principal Component 2',fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.tick_params(axis='both', which='minor', labelsize=14)

fig.tight_layout()
figdir = '/Users/mmdekker/Documents/Werk/Figures/Neurostuff/Pics_'+date+'/Gridding/'
if save==1: plt.savefig(figdir+string+'.png',bbox_inches = 'tight',dpi=200)