# ----------------------------------------------------------------- #
# About script
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import configparser
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import warnings
warnings.filterwarnings("ignore")
config = configparser.ConfigParser()
config.read('../Pipeline/config.ini')
Path_Data = config['PATHS']['DATA_RAW']
Path_Datasave = config['PATHS']['DATA_SAVE']
Path_Figsave = config['PATHS']['FIG_SAVE']

# ----------------------------------------------------------------- #
# Input
# ----------------------------------------------------------------- #

SUBs = ['279418', '279419', '339295', '339296', '346110', '346111']
REG = 'PFC'
REGs = ['HIP', 'PAR', 'PFC']
Name = REGs[0]+'_'+SUBs[0]+'.pkl'
Names = []
for REG in REGs:
    Namei = []
    for SUB in SUBs:
        Name = REG+'_'+SUB+'.pkl'
        Namei.append(Name)
    Names.append(Namei)
Names = np.array(Names)

# ----------------------------------------------------------------- #
# Create covariance matrices (not smoothed)
# ----------------------------------------------------------------- #

# Separate covariance matrices
CovMats = []
for i in range(len(Names)): # Regions
    CovMati = []
    for j in range(len(Names[0])): # Rodents
        Name = Names[i, j]
        Series = np.array(pd.read_pickle(Path_Datasave+'ResidualSeries/F_'+Name))
        Series[np.isnan(Series)] = 0
        CovMat = Series.dot(Series.T)
        CovMati.append(CovMat)
    CovMats.append(CovMati)
CovMats = np.array(CovMats)

# One averaged covariance matrix
AvCovMat = CovMats.mean(axis=1)

# ----------------------------------------------------------------- #
# Compute EOFs
# ----------------------------------------------------------------- #

# Not-smoothed, separate
EOFs_ind = np.zeros(shape=(3, 6, 2, 50))
for rod in range(6):
    for reg in range(3):
        vals, vecs = np.linalg.eig(CovMats[reg][rod])
        idx = vals.argsort()[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]
        vals_num = np.copy(vals/np.sum(vals))
        for i in range(2):
            EOFs_ind[reg, rod, i] = vecs[:, i]

# Not-smoothed, average
EOFs_av = np.zeros(shape=(3, 2, 50))
for reg in range(3):
    vals, vecs = np.linalg.eig(AvCovMat[reg])
    idx = vals.argsort()[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    vals_num = np.copy(vals/np.sum(vals))
    for i in range(2):
        EOFs_av[reg, i] = vecs[:, i]

#%%
# ----------------------------------------------------------------- #
# Calculate PCs, exploration data and rodent
# ----------------------------------------------------------------- #

st = ['', 'T']
Rodent = np.zeros(shape=(6789270))
PC1s = np.zeros(shape=(3, 6789270))
PC2s = np.zeros(shape=(3, 6789270))
Exploration = np.zeros(shape=(6789270))
for reg in tqdm(range(3)):
    EOF1 = EOFs_av[reg, 0]
    EOF2 = EOFs_av[reg, 1]
    expl = []
    rds = []
    pc1 = []
    pc2 = []
    for rod in range(6):
        for t in st:
            try:
                Name = Names[reg, rod][:-4]+t+'.pkl'
                Series = np.array(pd.read_pickle(Path_Datasave+'ResidualSeries/'+Name))
                Expl = np.array(pd.read_pickle(Path_Datasave+'Exploration/'+Name)).T[0]
                Expl = Expl[500:-500]
                rds = rds+[rod]*len(Expl)
                pc1 = pc1+list(EOF1.dot(Series))
                pc2 = pc2+list(EOF2.dot(Series))
                expl = expl+list(Expl)
            except:
                continue
    Rodent = np.array(rod)
    Exploration = np.array(expl)
    PC1s[reg] = np.array(pc1)
    PC2s[reg] = np.array(pc2)

#%%
# ----------------------------------------------------------------- #
# Smoothen PCs
# ----------------------------------------------------------------- #


def windowmean_adapted(Data, size):
    halfsize = int(size / 2)
    Result = np.zeros(len(Data))+np.nan
    for i in range(halfsize, np.int(len(Data)-halfsize)):
        Result[i] = np.nanmean(Data[i-halfsize:i+halfsize])
    return np.array(Result)


PC1s_sm = np.zeros(shape=(3, 6789270))
PC2s_sm = np.zeros(shape=(3, 6789270))
for reg in tqdm(range(3)):
    PC1s_sm[reg] = windowmean_adapted(PC1s[reg], 30)
    PC2s_sm[reg] = windowmean_adapted(PC2s[reg], 30)

# ----------------------------------------------------------------- #
# Save files
# ----------------------------------------------------------------- #



#%%
for i in range(4):
    if i == 0:
        vals, vecs = np.linalg.eig(covmats[i])
        idx = vals.argsort()[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]
        vals_num = np.copy(vals/np.sum(vals))
        EOFs_av = np.array([vecs[:, 0], vecs[:, 1]])
        EOFs[i, 0, 1] = vecs[:, 1]

#%%

# ----------------------------------------------------------------- #
# Read data and compute averages
# ----------------------------------------------------------------- #

config = configparser.ConfigParser()
config.read('../Pipeline/config.ini')
Path_Data = config['PATHS']['DATA_RAW']
Path_Datasave = config['PATHS']['DATA_SAVE']
Path_Figsave = config['PATHS']['FIG_SAVE']

EOFss = []
EVss = []
for i in range(len(SUBs)):
    Name = 'F_'+REG+'_'+SUBs[i]
    EOFs = np.array(pd.read_pickle(Path_Datasave+'EOFs/'+Name+'.pkl'))[:25]
    EVs = np.array(pd.read_pickle(Path_Datasave+'Eigenvalues/'+Name +
                                  '.pkl')).T[0]
    EOFss.append(EOFs)
    EVss.append(EVs)
Freqs = np.arange(1, 99+1, 2)
EOFss = np.array(EOFss)
for i in range(len(EOFss)):
    Averages = EOFss[:i].mean(axis=0)
    for j in range(len(EOFss[0])):
        r = np.corrcoef(Averages[j], EOFss[i][j])[0][1]
        if r < 0:
            EOFss[i][j] = -EOFss[i][j]
    Averages = EOFss[:i].mean(axis=0)
Av_final = EOFss.mean(axis=0)

# --------------------------------------------------------------------- #
# Determine and save average eigenvectors
# --------------------------------------------------------------------- #

DF = pd.DataFrame(Av_final)
DF.to_csv(Path_Datasave+'EOFs/Av_'+REG+'.csv')

#%%
# --------------------------------------------------------------------- #
# Determine and save averaged cov-matrix determined eigenvectors
# --------------------------------------------------------------------- #

st = ['', 'T']

for REG in tqdm(REGs):
    avcovmat = []
    for SUB in SUBs:
        for j in range(2):
            try:
                Name = REG+'_'+SUB+st[j]+'.pkl'
                dat = np.array(pd.read_pickle(Path_Datasave+'ResidualSeries/'+Name))
                covmat = dat.dot(dat.T)
                avcovmat.append(covmat)
            except:
                continue
    avcovmaty = np.nanmean(avcovmat,axis=0)
    pd.DataFrame(avcovmaty).to_csv(Path_Datasave+'ResidualSeries/AvCovMat_'+REG+'.csv')
    vals, vecs = np.linalg.eig(avcovmaty)
    idx = vals.argsort()[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    vals_num = np.copy(vals/np.sum(vals))
    EOFs = []
    for i in range(25):
        EOFs.append(vecs[:, i])
    pd.DataFrame(EOFs).to_csv(Path_Datasave+'EOFs/AvCovMat_'+REG+'.csv')

#%%
# --------------------------------------------------------------------- #
# Determine and save averaged cov-matrix determined eigenvectors
# --------------------------------------------------------------------- #

# Resid = [[]]*50
# for i in tqdm(range(len(SUBs))):
#     SUB = SUBs[i]
#     halfcut = len(np.array(pd.read_pickle(Path_Datasave+'Exploration/'+REG+'_'+SUB+'.pkl')).T[0])
#     ex = np.array(pd.read_pickle(Path_Datasave+'ResidualSeries/F_'+REG+'_'+SUB+'.pkl'))
#     for j in range(50):
#         Resid[j] = Resid[j]+list(ex[j, 500:halfcut-500])+list(ex[j, 500+halfcut:-500])
# Resid = np.array(Resid)
# AvCovMat = Resid.dot(Resid.T)
# pd.DataFrame(AvCovMat).to_csv(Path_Datasave+'ResidualSeries/TotCovMat_'+REG+'.csv')
# #%%
# vals, vecs = np.linalg.eig(AvCovMat)
# idx = vals.argsort()[::-1]
# vals = vals[idx]
# vecs = vecs[:, idx]
# vals_num = np.copy(vals/np.sum(vals))
# EOFs = []
# for i in range(25):#len(self.ResidualSeries)):
#     EOFs.append(vecs[:, i])
# pd.DataFrame(EOFs).to_csv(Path_Datasave+'EOFs/TotCovMat_'+REG+'.csv')

#%% Small check

Totmat = np.array(pd.read_csv(Path_Datasave+'ResidualSeries/TotCovMat_'+'PFC'+'.csv', index_col=0))
Avmat = np.array(pd.read_csv(Path_Datasave+'ResidualSeries/AvCovMat_'+'PFC'+'.csv', index_col=0))

Totmat = Totmat / np.max(Totmat)
Avmat = Avmat / np.max(Avmat)
plt.pcolormesh(range(0, 100, 2), range(0, 100, 2), np.abs(Totmat-Avmat)/np.abs(Avmat), vmin=0, vmax=0.25)
plt.colorbar(label='Fractional difference'+'\n'+'between avcovmat and totcovmat')
plt.ylabel('Frequency')
plt.xlabel('Frequency')
#%%
# --------------------------------------------------------------------- #
# Get projected time series ('PCs')
# --------------------------------------------------------------------- #

EOFs_hip = np.array(pd.read_csv(Path_Datasave+'EOFs/AvCovMat_HIP.csv', index_col=0))[:2]
EOFs_pfc = np.array(pd.read_csv(Path_Datasave+'EOFs/AvCovMat_PFC.csv', index_col=0))[:2]
EOFs_par = np.array(pd.read_csv(Path_Datasave+'EOFs/AvCovMat_PAR.csv', index_col=0))[:2]

TS = []
EX = []
SU = []
EOFs = [EOFs_hip, EOFs_par, EOFs_pfc]
for j in tqdm(range(len(REGs))):
    REG = REGs[j]
    for e in range(2):
        eof = EOFs[j][e]
        Ts_reg_eof = []
        Expl = []
        Sub = []
        for i in range(len(SUBs)):
            SUB = SUBs[i]
            halfcut = len(np.array(pd.read_pickle(Path_Datasave+'Exploration/'+REG+'_'+SUB+'.pkl')).T[0])
            Ts_new = eof.dot(pd.read_pickle(Path_Datasave+'ResidualSeries/F_'+REG+'_'+SUB+'.pkl'))
            Ts_add = np.array(list(Ts_new[:halfcut])+[np.nan]*30+list(Ts_new[halfcut:])+[np.nan]*30)
            Ex_new = np.array(pd.read_pickle(Path_Datasave+'Exploration/F_'+REG+'_'+SUB+'.pkl')).T[0]
            Ex_add = np.array(list(Ex_new[500:halfcut-500])+[np.nan]*30+list(Ex_new[500+halfcut:-500])+[np.nan]*30)
            Ts_reg_eof = Ts_reg_eof+list(Ts_add)
            Expl = Expl+list(Ex_add)
            Sub = Sub+[i]*len(Ex_add)
        TS.append(Ts_reg_eof)
        EX.append(Expl)
        SU.append(Sub)
TS = np.array(TS)
EX = np.array(EX)
SU = np.array(SU)
#%%
# ----------------------------------------------------------------- #
# Save data to dataframe
# ----------------------------------------------------------------- #

DF = {}
DF['HIP1'] = TS[0]
DF['HIP2'] = TS[1]
DF['PAR1'] = TS[2]
DF['PAR2'] = TS[3]
DF['PFC1'] = TS[4]
DF['PFC2'] = TS[5]
DF['Expl'] = EX[0]
DF['ROD'] = SU[0]

pd.DataFrame(DF).to_pickle(Path_Datasave+'PCs/Total.pkl')