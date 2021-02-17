# ----------------------------------------------------------------- #
# About script
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import configparser
import sys
import scipy.io
import networkx as nx
from community import community_louvain
from community import modularity
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Functions import Standardize, RemovePowerlaw2b, Abundances, ProcessPower
from Functions import func12, func21
from scipy.fftpack import rfftfreq, rfft
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Class object
# --------------------------------------------------------------------------- #


class Brain(object):
    def __init__(self, params_input):
        ''' Initial function '''

        config = configparser.ConfigParser()
        config.read('config.ini')

        ''' Dataset '''
        self.SUBJECT = params_input['SUBJECT']
        self.TYPE = params_input['TYPE']
        self.REGION = params_input['REGION']
        self.FULL = int(config['DATASET']['FULL'])

        ''' Important constants '''
        self.N = int(config['PARAMS']['N'])
        self.F_LOW = int(config['PARAMS']['F_LOW'])
        self.F_HIGH = int(config['PARAMS']['F_HIGH'])
        self.SUBPART = params_input['SUBPART']
        self.WS = int(config['PARAMS']['WINDOW'])
        self.TAU = int(config['PARAMS']['TAU'])
        self.RES = int(config['PARAMS']['RES'])
        self.PCX = int(config['PARAMS']['PCX'])
        self.PCY = int(config['PARAMS']['PCY'])

        ''' Paths '''
        self.Path_Data = config['PATHS']['DATA_RAW']
        self.Path_Datasave = config['PATHS']['DATA_SAVE']
        self.Path_Figsave = config['PATHS']['FIG_SAVE']

        ''' Read dataset '''
        self.RawData = scipy.io.loadmat(self.Path_Data+self.SUBJECT+'_' +
                                        self.TYPE+'.mat')
        if self.TYPE == 'TRAINING' and self.SUBJECT == '340233':
            D = scipy.io.loadmat(self.Path_Data+'Cat_340223_Training.mat')
            self.Datacat = D['interpolatedCategory']
        else:
            self.Datacat = self.RawData['interpolatedCategory']
        self.DataEEG = self.RawData['EEG'][0][0][9]  # 15 before
        if self.N == -1:
            self.N = len(self.Datacat[0])
        if self.TYPE == 'TEST':
            self.SUBJECT += 'T'

        print('# ---------------------------------------------------------- #')
        print('# Starting calculations for '+self.SUBJECT+'-'+self.TYPE +
              ' in '+self.REGION+' '+str(self.N)+' ('+str(self.SUBPART)+')')
        print('# ---------------------------------------------------------- #')

        if self.REGION == 'PFC':
            self.NODES = range(0, 16)
        if self.REGION == 'PAR':
            self.NODES = range(16, 24)
        if self.REGION == 'HIP':
            self.NODES = range(24, 32)
        if self.SUBPART == 'NO':
            self.Datacat = self.Datacat[:self.N][0]
            self.DataEEG = self.DataEEG[self.NODES, :self.N]
        else:
            self.SUBPART = int(self.SUBPART)
            self.Datacat = self.Datacat[:self.N][0]
            self.DataEEG = self.DataEEG[self.NODES,
                                        self.N*self.SUBPART:self.N *
                                        self.SUBPART+self.N]

    def preprocess(self):
        ''' Preprocess raw data '''

        self.SEEG = Standardize(self.DataEEG)

    def determine_residual_series(self):
        ''' Perform Fourier transform, power-law fit and calculate
            residuals '''
        
        if self.FULL == 0:
            ''' Perform one iterations, to find the correct borders '''
            one_iter = self.SEEG[:, 0:2*self.WS]
            freq_red = rfftfreq(len(one_iter[0]), d=1e-3)
            Hz1 = np.where(freq_red >= self.F_LOW-0.5)[0][0]
            Hz2 = np.where(freq_red <= self.F_HIGH+0.5)[0][-1]
            frq = freq_red[Hz1:Hz2]
            borders = np.arange(self.F_LOW, self.F_HIGH+0.5, 2)
            freq_ind = []
            ran = np.arange(len(frq))
            for i in range(len(borders)-1):
                freq_ind.append(ran[(frq > borders[i]) & (frq <= borders[i+1])])
    
            ''' Start iterating over all windows, determine FFT and remove
                the power laws '''
            RelPower_region = []
            synchr_region = []
            tvec = np.arange(self.WS, self.N-self.WS)
            A = np.zeros(shape=(len(tvec), len(self.SEEG)))
            B = np.zeros(shape=(len(tvec), len(self.SEEG)))
            for stepi in tqdm(range(len(tvec)), file=sys.stdout):
                step = tvec[stepi]
                Data_step = self.SEEG[:, step-self.WS:step+self.WS]
                RelPower_node = []
                ffts = []
                As = []
                Bs = []
    
                ''' Per node in this brain region '''
                for node in range(len(Data_step)):
    
                    ''' FFT and normalize it '''
                    FFT = np.abs(rfft(Data_step[node]))/len(Data_step[node])*2
                    FFT = FFT/np.nansum(FFT)
    
                    ''' Remove power law '''
                    a, b, FFTn = RemovePowerlaw2b(FFT, frq, Hz1, Hz2)
                    ffts.append(FFTn)
                    A[stepi, node] = a
                    B[stepi, node] = b
    
                    ''' Calculate relative power, binned per Hz-segment '''
                    RelPower_node.append(Abundances(FFTn, frq, borders, freq_ind))
                synchr_region.append(np.nanmean(np.corrcoef(ffts)))
                RelPower_region.append(np.nanmean(RelPower_node, axis=0))
            self.A = A
            self.B = B
            self.Synchronization = np.array(synchr_region)
            self.ResidualSeries_raw = np.array(RelPower_region)
            self.ResidualSeries = ProcessPower(self.ResidualSeries_raw)
            self.Datacat[self.WS:self.N-self.WS-1]
        elif self.SUBJECT != '339296': # Exception when no test data is available
            name = self.REGION+'_'+self.SUBJECT
            Sync1 = np.array(pd.read_pickle(self.Path_Datasave+'Synchronization/'+name+'.pkl'))
            Sync2 = np.array(pd.read_pickle(self.Path_Datasave+'Synchronization/'+name+'T.pkl'))
            self.Synchronization = np.array(list(Sync1)+list(Sync2)).T[0]

            Cat1 = np.array(pd.read_pickle(self.Path_Datasave+'Exploration/'+name+'.pkl'))
            Cat2 = np.array(pd.read_pickle(self.Path_Datasave+'Exploration/'+name+'T.pkl'))
            self.Datacat = np.array(list(Cat1)+list(Cat2)).T[0]

            Res1 = np.array(pd.read_pickle(self.Path_Datasave+'ResidualSeries/'+name+'.pkl'))
            Res2 = np.array(pd.read_pickle(self.Path_Datasave+'ResidualSeries/'+name+'T.pkl'))
            self.ResidualSeries = []
            for i in range(len(Res1)):
                self.ResidualSeries.append(list(Res1[i])+list(Res2[i]))
            self.ResidualSeries = np.array(self.ResidualSeries)
        else:
            name = self.REGION+'_'+self.SUBJECT
            Sync1 = np.array(pd.read_pickle(self.Path_Datasave+'Synchronization/'+name+'.pkl'))
            self.Synchronization = np.array(list(Sync1)).T[0]

            Cat1 = np.array(pd.read_pickle(self.Path_Datasave+'Exploration/'+name+'.pkl'))
            self.Datacat = np.array(list(Cat1)).T[0]

            Res1 = np.array(pd.read_pickle(self.Path_Datasave+'ResidualSeries/'+name+'.pkl'))
            self.ResidualSeries = []
            for i in range(len(Res1)):
                self.ResidualSeries.append(list(Res1[i]))
            self.ResidualSeries = np.array(self.ResidualSeries)

    def pca(self):
        ''' Perform Principal Component Analysis '''

        ''' Perform eigendecomposition '''
        vals, vecs = np.linalg.eig(self.ResidualSeries.dot(self.ResidualSeries.T))
        idx = vals.argsort()[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]
        vals_num = np.copy(vals/np.sum(vals))

        ''' Put EOFs and PCs in arrays '''
        EOFs = []
        PCs = []
        for i in range(25):#len(self.ResidualSeries)):
            EOFs.append(vecs[:, i])
            PCs.append(vecs[:, i].dot(self.ResidualSeries))

        self.EOFs = EOFs
        self.PCs = PCs
        self.Eigenvalues = vals_num

    def grid_rate_matrix(self):
        ''' Acquire grid-rate matrix including conditional probabilities '''

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
        T1 = 0
        T2 = len(Xpc)-self.TAU
        xe = np.array([xedges2]*len(Ypc[T1:T2]))
        ye = np.array([yedges2]*len(Xpc[T1:T2]))
        Locations = []
        for j in [0, self.TAU]:
            xe_arg = np.abs(xe.T-Xpc[T1+j:T2+j]).argmin(axis=0)
            ye_arg = np.abs(ye.T-Ypc[T1+j:T2+j]).argmin(axis=0)
            Locations.append(ye_arg*len(xedges2)+xe_arg)
        Locations = np.array(Locations)

        ''' Determine Rates matrix '''
        Rates = np.zeros(shape=(self.RES2**2, self.RES2**2))
        for i in range(len(Locations[0])):
            Rates[Locations[0, i], Locations[1, i]] += 1
        Rates_n = []
        for i in range(len(Rates)):
            Rates_n.append(np.copy(Rates[i])/np.nansum(Rates[i]))
        self.RatesMat = np.array(np.nan_to_num(Rates_n, 0))

    def clustering(self):
        ''' Cluster grid-rate matrix and determine its properties '''

        ''' Generate network object '''
        ConMat = self.RatesMat+self.RatesMat.T
        vec = np.array(np.where(np.sum(ConMat, axis=0) > 0)[0])
        AdjMat = self.RatesMat[vec][:, vec]
        edges = []
        weights = []
        for i in range(0, len(AdjMat)):
            for j in range(i+1, len(AdjMat)):
                if i != j and np.mean([AdjMat[i, j], AdjMat[j, i]]) > 0:
                    edges.append((i, j))
                    weights.append(np.mean([AdjMat[i, j], AdjMat[j, i]]))
        Networkx = nx.Graph()
        Networkx.add_nodes_from(np.unique(edges))
        Networkx.add_edges_from(edges)

        ''' Apply clustering algorithm '''
        partition = community_louvain.best_partition(Networkx, weight='weight')
        self.Modularity = modularity(partition, Networkx, weight='weight')

        ''' Obtain clusters '''
        count = 0.
        clusters = []
        for com in set(partition.values()):
            count = count+1.
            list_nodes = [nodes for nodes in partition.keys()
                          if partition[nodes] == com]
            clusters.append(vec[list_nodes])

        ''' Throw away very small clusters '''
        Clusters = []
        for i in range(len(clusters)):
            if len(clusters[i]) > 15:
                Clusters.append(clusters[i])
        Existence = []
        for i in range(len(Clusters)):
            vec = func21(np.zeros(shape=(self.RES2, self.RES2)))+np.nan
            vec[Clusters[i]] = 1
            Existence.append(vec)

        ''' Put clusters in Dataframe '''
        DF = {}
        DF['Cell'] = np.arange(len(self.RatesMat))
        clus = np.zeros(len(DF['Cell']))+np.nan
        for i in range(len(Clusters)):
            for j in Clusters[i]:
                clus[j] = int(i)
        DF['Cluster'] = clus
        self.ClusterDF = DF

    def get_from_file(self):
        ''' Get residual series from previous files '''

        if self.N > 500000:
            Run = self.REGION+'_'+self.SUBJECT
        else:
            Run = ('Sub_'+str(self.N)+'/'+self.REGION+'_'+self.SUBJECT+'_' +
                   str(self.N)+'_'+str(self.SUBPART))
        if self.FULL == 1:
            Run = 'F_'+Run
        self.ResidualSeries = np.array(pd.read_pickle(self.Path_Datasave +
                                                      'ResidualSeries/'+Run +
                                                      '.pkl'))
        # self.Datacat = np.array(pd.read_pickle(self.Path_Datasave +
        #                                        'Exploration/'+Run +
        #                                        '.pkl'))

    def save_to_file(self):
        ''' Save all important variables for future use '''

        if self.N > 500000:
            Run = self.REGION+'_'+self.SUBJECT
        else:
            Run = ('Sub_'+str(self.N)+'/'+self.REGION+'_'+self.SUBJECT+'_' +
                   str(self.N)+'_'+str(self.SUBPART))
        if self.FULL == 1:
            Run = 'F_'+Run
        try:
            pd.DataFrame(self.ClusterDF).to_pickle(self.Path_Datasave +
                                                   'Clusters/'+Run+'.pkl')
        except:
            3
        #pd.DataFrame(self.EOFs).to_pickle(self.Path_Datasave +
        #                                  'EOFs/'+Run+'.pkl')
        #pd.DataFrame(self.PCs).to_pickle(self.Path_Datasave +
        #                                 'PCs/'+Run+'.pkl')
        pd.DataFrame(self.Synchronization).to_pickle(self.Path_Datasave +
                                                     'Synchronization/'+Run +
                                                     '.pkl')
        pd.DataFrame(self.Synchronization).to_pickle(self.Path_Datasave +
                                                     'Synchronization/'+Run +
                                                     '.pkl')
        #pd.DataFrame(self.Eigenvalues).to_pickle(self.Path_Datasave +
        #                                         'Eigenvalues/'+Run+'.pkl')
        pd.DataFrame(self.ResidualSeries).to_pickle(self.Path_Datasave +
                                                    'ResidualSeries/'+Run +
                                                    '.pkl')
        pd.DataFrame(self.Datacat).to_pickle(self.Path_Datasave +
                                             'Exploration/'+Run+'.pkl')
        if self.FULL == 0:
            pd.DataFrame(self.A).to_csv(self.Path_Datasave+'A/'+Run+'.csv')
            pd.DataFrame(self.B).to_csv(self.Path_Datasave+'B/'+Run+'.csv')
