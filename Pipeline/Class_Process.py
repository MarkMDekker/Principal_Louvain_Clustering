# ----------------------------------------------------------------- #
# About script
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import configparser
import networkx as nx
import sys
import scipy.io
from community import community_louvain
from community import modularity
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Functions import Standardize, RemovePowerlaw2, Abundances, ProcessPower
from Functions import func12, func21, overlap_metric
from scipy.fftpack import rfftfreq, rfft
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Class object
# --------------------------------------------------------------------------- #


class Brain_process(object):
    def __init__(self, params_input):
        ''' Initial function '''

        config = configparser.ConfigParser()
        config.read('config_process.ini')

        ''' Dataset '''
        self.SUBJECT = config['DATASET']['SUBJECT']
        self.REGION = config['DATASET']['REGION']
        self.FULL = int(config['DATASET']['FULL'])
        self.ROBUST = int(config['DATASET']['ROBUST'])

        ''' Important constants '''
        self.SUBN = int(config['PARAMS']['SUBN'])
        self.TAU = int(config['PARAMS']['TAU'])
        self.RES = int(config['PARAMS']['RES'])
        self.PCX = int(config['PARAMS']['PCX'])
        self.PCY = int(config['PARAMS']['PCY'])
        self.W = int(config['PARAMS']['WINDOW'])
        self.TH = np.float(config['PARAMS']['NETW_TH'])
        self.MINCLUS = int(config['PARAMS']['MINCLUS'])

        ''' Paths '''
        self.Path_Data = config['PATHS']['DATA_RAW']
        self.Path_Datasave = config['PATHS']['DATA_SAVE']
        self.Path_Figsave = config['PATHS']['FIG_SAVE']

        ''' Read dataset '''
        self.RawData1 = scipy.io.loadmat(self.Path_Data+self.SUBJECT+'_Training.mat')
        self.RawData2 = scipy.io.loadmat(self.Path_Data+self.SUBJECT+'_Test.mat')
        Datacat1 = list(np.array(self.RawData1['interpolatedCategory'])[0][self.W:-self.W])
        Datacat2 = list(np.array(self.RawData2['interpolatedCategory'])[0][self.W:-self.W])
        self.Datacat = np.array(Datacat1+Datacat2)

        print('# ----------------------------------------------------------'
              '------------------- #')
        print('# Starting processing for '+self.SUBJECT +
              ' in '+self.REGION)
        print('# ----------------------------------------------------------'
              '------------------- #')

        self.Name_M = self.REGION+'_'+self.SUBJECT
        if self.FULL == 1:
            self.Name_M = 'F_'+self.Name_M

    def load_files(self):
        ''' loading necessary files '''

        st = self.Path_Datasave+'Clusters/'
        ClustersH = []
        if self.ROBUST == 1:
            for j in range(6):
                name = ('/Sub_'+str(self.SUBN)+'/'+self.REGION+'_'+self.SUBJECT +
                        '_'+str(self.SUBN))
                ClustersH.append(np.array(pd.read_pickle(st+name+'_' +
                                                         str(j)+'.pkl')))
            for j in range(6):
                name = ('/Sub_'+str(self.SUBN)+'/'+self.REGION+'_'+self.SUBJECT +
                        'T_'+str(self.SUBN))
                ClustersH.append(np.array(pd.read_pickle(st+name+'_' +
                                                         str(j)+'.pkl')))
        self.SubclusH = np.array(ClustersH)
        self.MasclusH = np.array(pd.read_pickle(st+self.Name_M+'.pkl'))
        self.PCs = np.array(pd.read_pickle(self.Path_Datasave+'PCs/' +
                                           self.Name_M+'.pkl'))

    def select_master_clusters(self):
        ''' select master clusters that are robust enough '''

        ''' Create similarity network '''
        setlist = []
        Clusters = []
        for i in range(len(self.SubclusH)):
            for j in range(int(np.nanmax(self.SubclusH[i].T[1])+1)):
                Clusters.append(np.where(self.SubclusH[i].T[1] == j)[0])
                setlist.append(i)
        setlist = np.array(setlist)

        MasterClusters = []
        for j in range(int(np.nanmax(self.MasclusH.T[1])+1)):
            MasterClusters.append(np.where(self.MasclusH.T[1] == j)[0])

        SimMat = np.zeros(shape=(len(MasterClusters), len(Clusters)))
        for i in range(len(MasterClusters)):
            for j in range(len(Clusters)):
                SimMat[i, j] = overlap_metric(MasterClusters[i], Clusters[j])

        ss = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                       'L', 'M', 'N'])
        edgesraw = np.array(np.where(SimMat > self.TH))
        edges = np.array([ss[edgesraw[0]], edgesraw[1]]).T
        weights = SimMat[np.where(SimMat > self.TH)]
        self.Subclus = Clusters
        self.Masclus = MasterClusters
        self.Network = nx.Graph()
        self.Network.add_edges_from(edges)
        self.Network.add_nodes_from(ss[np.arange(len(self.Masclus))])
        self.Netw_weights = weights
        self.Netw_setlist = setlist
        self.Netw_edges = edges

        ''' Apply Louvain clustering '''
        partition = community_louvain.best_partition(self.Network,
                                                     weight='weight')
        keys = np.array(list(partition.keys()))
        vals = np.array(list(partition.values()))
        weight_clus = []
        Clusclus = []
        sets = []
        for i in range(np.max(vals)+1):
            ke = keys[np.where(vals == i)[0]]
            if len(ke) >= 1:
                ke2 = [ke[0]]
                for K in ke:
                    try:
                        ke2.append(int(K))
                    except:
                        continue
                Clusclus.append(ke2)
                sets.append(setlist[ke2[1:]])
                char = ke2[1:]
                weight_clus.append(weights[np.where(edges.T[0] == char)])

        ''' Select Louvain clusters and associated subclusters '''
        self.MasClus_s = []
        self.TotClus_s = []
        self.WeiClus_s = []
        for i in range(len(Clusclus)):         
            if len(np.unique(sets[i])) >= self.MINCLUS:
                self.MasClus_s.append(Clusclus[i][0])
                self.TotClus_s.append(Clusclus[i])
                self.WeiClus_s.append(weight_clus[i])

    def plot_preselection_network(self):
        ''' plot graph of master clusters and subclusters '''

        dicty = {}
        for i in range(len(self.Netw_edges)):
            dicty[tuple(self.Netw_edges[i])] = np.round(self.Netw_weights[i],
                                                        2)
        colors = ['forestgreen', 'tomato', 'steelblue', 'orange', 'violet',
                  'brown', 'navy', 'dodgerblue', 'goldenrod', 'magenta', 'grey', 'green']
        cols = []
        for i in self.Network.nodes:
            try:
                cols.append(colors[self.Netw_setlist[int(i)]])
            except:
                cols.append('silver')
        edgeweights = []
        for u, v in self.Network.edges():
            try:
                edgeweights.append(dicty[u, v])
            except KeyError:
                edgeweights.append(dicty[v, u])
        edgeweights = np.array(edgeweights)
        pos = nx.nx_agraph.graphviz_layout(self.Network, prog='neato')
        fig, ax = plt.subplots(figsize=(15, 15))
        nx.draw(self.Network, pos, node_size=1000, node_color=cols,
                width=0.1+5*edgeweights**2)
        nx.draw_networkx_labels(self.Network, pos, font_size=15)
        nx.draw_networkx_edge_labels(self.Network, pos, edge_labels=dicty)
        ax.text(0.0, 0.95+1*0.03, 'Sets of '+str(int(self.SUBN/1000))+' sec:',
                color='k', transform=ax.transAxes, fontsize=15)
        for i in np.unique(self.Netw_setlist):
            ax.text(0.02, 0.95-i*0.02, 'Set '+str(i), color=colors[i],
                    fontsize=15, transform=ax.transAxes)
        ax.text(0.98, 0.02, 'Threshold of '+str(self.TH), color='k',
                fontsize=15, transform=ax.transAxes, ha='right', va='bottom')

    def acquire_locations(self):
        ''' Create space '''
        Xpc = self.PCs[self.PCX]
        Ypc = self.PCs[self.PCY]
        self.space = np.linspace(-12, 12, self.RES)

        ''' Determine histogram '''
        H, xedges, yedges = np.histogram2d(Xpc, Ypc, bins=[self.space,
                                                           self.space])
        xedges2 = (xedges[:-1] + xedges[1:])/2.
        yedges2 = (yedges[:-1] + yedges[1:])/2.
        self.RES2 = len(xedges2)

        ''' Determine Locations '''
        xe = np.array([xedges2]*len(Ypc))
        ye = np.array([yedges2]*len(Xpc))
        xe_arg = np.abs(xe.T-Xpc).argmin(axis=0)
        ye_arg = np.abs(ye.T-Ypc).argmin(axis=0)
        Locations = np.array(ye_arg*len(xedges2)+xe_arg)
        self.Locations = Locations

    def save_processing(self):
        ''' loading necessary files '''

        ss = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                       'L', 'M', 'N'])
        clusjes = self.MasclusH[self.Locations].T[1].astype(int)
        Cluster_series = np.zeros(len(clusjes))+np.nan
        Cluster_series = Cluster_series.astype(str)
        Cluster_series[clusjes >= 0] = ss[clusjes[clusjes >= 0]]
        dicty = {}
        dicty['Act'] = self.Datacat
        dicty[self.REGION] = Cluster_series
        DF = pd.DataFrame(dicty)
        DF[self.REGION][~DF[self.REGION].isin(ss)] = '-'  # Cover all ms
        self.DF = DF
        self.DF.to_pickle(self.Path_Datasave+'ActiveClusters/'+self.Name_M+'.pkl')
        pd.DataFrame(self.Locations).to_pickle(self.Path_Datasave+'Locations/'+self.Name_M+'.pkl')
