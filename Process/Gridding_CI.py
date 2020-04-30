# --------------------------------------------------------------------- #
# Preambule
# --------------------------------------------------------------------- #

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from community import community_louvain
from community import modularity
path_pdata = ('/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/'
              'ProcessedData/')

# --------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------- #

Str = 'HIP'
N = 100000
for prt in tqdm(range(5)):
    s = Str+'/'+str(N)+'_'+str(prt)

# --------------------------------------------------------------------- #
# Read data
# --------------------------------------------------------------------- #

    PCs = np.array(pd.read_pickle(path_pdata+'PCs/'+s+'.pkl'))
    Xpc = PCs[0]
    Ypc = PCs[1]
    tau = 30
    res = 200

# --------------------------------------------------------------------- #
# Retrieve phase-space, calculate locations and rates
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

    for i in range(len(Locations[0])):
        Rates[Locations[0, i], Locations[1, i]] += 1

    Rates_n = []
    for i in range(len(Rates)):
        Rates_n.append(np.copy(Rates[i])/np.nansum(Rates[i]))

    Rates_n = np.array(np.nan_to_num(Rates_n, 0))

    # --------------------------------------------------------------------- #
    # Create and perform clustering (Louvain)
    # --------------------------------------------------------------------- #

    def LouvainClustering(Ratesmat):
        ConMat = Ratesmat+Ratesmat.T
        vec = np.array(np.where(np.sum(ConMat, axis=0) > 0)[0])
        AdjMat = Ratesmat[vec][:, vec]
        nodes = range(0,len(AdjMat))
        edges = []
        weights = []
        for i in range(0,len(AdjMat)):
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
    
    # --------------------------------------------------------------------- #
    # Apply clustering algorithm
    # --------------------------------------------------------------------- #
    
    clusters, mod = LouvainClustering(Rates)
    res2 = len(Xvec)
    
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
    
    # --------------------------------------------------------------------- #
    # Save clusters
    # --------------------------------------------------------------------- #
    
    DF = {}
    DF['Cell'] = np.arange(len(Rates_n))
    clus = np.zeros(len(DF['Cell']))+np.nan
    for i in range(len(Clusters)):
        for j in Clusters[i]:
            clus[j] = int(i)
    DF['Cluster'] = clus
    
    pd.DataFrame(DF).to_csv(path_pdata+'Clusters/'+s+'.csv', index=False)
