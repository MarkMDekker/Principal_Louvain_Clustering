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
from Functions import Standardize, RemovePowerlaw3, Abundances, ProcessPower, RemovePowerlaw2b, RemovePowerlaw2a
from Functions import func12, func21
from scipy.fftpack import rfftfreq, rfft
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Class object
# --------------------------------------------------------------------------- #


class Tester(object):
    def __init__(self, params_input):
        ''' Initial function '''

        config = configparser.ConfigParser()
        config.read('config_TestAB.ini')

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
            freq_ind.append(ran[(frq > borders[i]
                                 ) & (frq <= borders[i+1])])

        ''' Start iterating over all windows, determine FFT and remove
            the power laws '''
        tvec = np.arange(self.WS, self.N-self.WS)[:1000]
        A = np.zeros(shape=(len(tvec), len(self.SEEG)))
        B = np.zeros(shape=(len(tvec), len(self.SEEG)))
        C = np.zeros(shape=(len(tvec), len(self.SEEG)))
        RelPower_region = []
        for stepi in tqdm(range(len(tvec)), file=sys.stdout):
            step = tvec[stepi]
            Data_step = self.SEEG[:, step-self.WS:step+self.WS]
            RelPower_node = []

            ''' Per node in this brain region '''
            for node in range(len(Data_step)):

                ''' FFT and normalize it '''
                FFT = np.abs(rfft(Data_step[node]))/len(Data_step[node])*2
                FFT = FFT/np.nansum(FFT)

                ''' Remove power law '''
                #a, b, c, FFTn = RemovePowerlaw3(FFT, frq, Hz1, Hz2)
                a, b, FFTn = RemovePowerlaw2b(FFT, frq, Hz1, Hz2)
                A[stepi, node] = a
                B[stepi, node] = b
                #C[stepi, node] = c
                RelPower_node.append(Abundances(FFTn, frq, borders, freq_ind))
            RelPower_region.append(np.nanmean(RelPower_node, axis=0))
        self.A = A
        self.B = B
        #self.C = C
        self.ResidualSeries_raw = np.array(RelPower_region)
        self.ResidualSeries = ProcessPower(self.ResidualSeries_raw)

    def save_to_file(self):
        ''' Save all important variables for future use '''

        Run = self.REGION+'_'+self.SUBJECT
        pd.DataFrame(self.A).to_csv(self.Path_Datasave+'A/'+Run+'.csv')
        pd.DataFrame(self.B).to_csv(self.Path_Datasave+'B/'+Run+'.csv')