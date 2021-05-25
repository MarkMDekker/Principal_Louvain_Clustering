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

SUBs = ['279418', '279419', '339295', '339296', '340233', '341319',
        '341321', '346110', '346111', '347640', '347641']
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
for i in tqdm(range(len(Names))):  # Regions
    CovMati = []
    for j in range(len(Names[0])):  # Rodents
        Name = Names[i, j]
        Series = np.array(pd.read_pickle(Path_Datasave+'ResidualSeries/F_' +
                                         Name))
        Series[np.isnan(Series)] = 0
        CovMat = Series.dot(Series.T)
        CovMati.append(CovMat)
    CovMats.append(CovMati)
CovMats = np.array(CovMats)

# One averaged covariance matrix
AvCovMat = CovMats.mean(axis=1)

#%%
# ----------------------------------------------------------------- #
# Compute EOFs
# ----------------------------------------------------------------- #

# Not-smoothed, separate
Vals_ind = []
EOFs_ind = np.zeros(shape=(3, len(SUBs), 2, 50))
for rod in range(len(SUBs)):
    valind = []
    for reg in range(3):
        vals, vecs = np.linalg.eig(CovMats[reg][rod])
        idx = vals.argsort()[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]
        vals_num = np.copy(vals/np.sum(vals))
        valind.append(vals_num)
        for i in range(2):
            EOFs_ind[reg, rod, i] = vecs[:, i]
    Vals_ind.append(valind)

# Not-smoothed, average
EOFs_av = np.zeros(shape=(3, 2, 50))
Vals = []
for reg in range(3):
    vals, vecs = np.linalg.eig(AvCovMat[reg])
    idx = vals.argsort()[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    vals_num = np.copy(vals/np.sum(vals))
    Vals.append(vals_num)
    for i in range(2):
        EOFs_av[reg, i] = vecs[:, i]
DF_vals = {}
DF_vals['HIP'] = Vals[0]
DF_vals['PAR'] = Vals[1]
DF_vals['PFC'] = Vals[2]
for rod in range(len(SUBs)):
    for reg in range(len(REGs)):
        nam = SUBs[rod]+'_'+REGs[reg]
        DF_vals[nam] = Vals_ind[rod][reg]
DF_vals = pd.DataFrame(DF_vals)
DF_vals.to_csv('/Users/mmdekker/Documents/Werk/Data/Neuro/Processed/Spectra.csv')

#%%
# ----------------------------------------------------------------- #
# Calculate PCs, exploration data and rodent
# ----------------------------------------------------------------- #

tot = 13644532
st = ['', 'T']
Rodent = np.zeros(shape=(tot))
PC1s = np.zeros(shape=(3, tot))
PC2s = np.zeros(shape=(3, tot))
Exploration = np.zeros(shape=(tot))
for reg in tqdm(range(3)):
    EOF1 = EOFs_av[reg, 0]
    EOF2 = EOFs_av[reg, 1]
    expl = []
    rds = []
    pc1 = []
    pc2 = []
    for rod in range(len(SUBs)):
        for t in st:
            Name = Names[reg, rod][:-4]+t+'.pkl'
            Series = np.array(pd.read_pickle(Path_Datasave+'ResidualSeries/'+Name))
            Expl = np.array(pd.read_pickle(Path_Datasave+'Exploration/'+Name)).T[0]
            Expl = Expl[500:-500]
            rds = rds+[rod]*len(Expl)
            pc1 = pc1+list(EOF1.dot(Series))
            pc2 = pc2+list(EOF2.dot(Series))
            expl = expl+list(Expl)
    Rodent = np.array(rds)
    Exploration = np.array(expl)
    PC1s[reg] = np.array(pc1)
    PC2s[reg] = np.array(pc2)

#%%
# ----------------------------------------------------------------- #
# Smoothen PCs
# ----------------------------------------------------------------- #


def conv(datx, size=30):
    y = np.exp(-(np.arange(30)-14.5)**2/(2*12**2))/23.72772166375371
    return np.dot(y, datx)


def windowmean_adapted(Data, size):
    halfsize = int(size / 2)
    Result = np.zeros(len(Data))+np.nan
    for i in range(halfsize, np.int(len(Data)-halfsize)):
        dat = Data[i-halfsize:i+halfsize]
        Result[i] = conv(dat)
    return np.array(Result)


def windowmean_std(Data, size):
    halfsize = int(size / 2)
    Result = np.zeros(len(Data))+np.nan
    for i in range(halfsize, np.int(len(Data)-halfsize)):
        dat = Data[i-halfsize:i+halfsize]
        Result[i] = np.mean(dat)
    return np.array(Result)


PC1s_sm = np.zeros(shape=(3, tot))
PC2s_sm = np.zeros(shape=(3, tot))
for reg in tqdm(range(3)):
    PC1s_sm[reg] = windowmean_adapted(PC1s[reg], 30)
    PC2s_sm[reg] = windowmean_adapted(PC2s[reg], 30)

PC1s_shuf = np.copy(PC1s_sm)
PC2s_shuf = np.copy(PC2s_sm)
newidx = np.random.choice(range(tot), size=tot, replace=False)
for reg in tqdm(range(3)):
    PC1s_shuf[reg] = PC1s_shuf[reg][newidx]
    PC2s_shuf[reg] = PC2s_shuf[reg][newidx]
#%%
# ----------------------------------------------------------------- #
# Save files
# ----------------------------------------------------------------- #

savepath = '/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/Processed/'

for reg in range(3):
    REG = REGs[reg]

    # EOF1
    DF_eof1 = {}
    DF_eof1['Average'] = EOFs_av[reg][0]
    for rod in range(len(SUBs)):
        SUB = SUBs[rod]
        eof = EOFs_ind[reg][rod][0]
        if np.corrcoef(eof, EOFs_av[reg][0])[0][1] < 0:
            eof = -eof
        DF_eof1[SUB] = eof
    pd.DataFrame(DF_eof1).to_csv(savepath+'EOF1/'+REG+'.csv')

    # EOF2
    DF_eof2 = {}
    DF_eof2['Average'] = EOFs_av[reg][1]
    for rod in range(len(SUBs)):
        SUB = SUBs[rod]
        eof = EOFs_ind[reg][rod][1]
        if np.corrcoef(eof, EOFs_av[reg][1])[0][1] < 0:
            eof = -eof
        DF_eof2[SUB] = eof
    pd.DataFrame(DF_eof2).to_csv(savepath+'EOF2/'+REG+'.csv')

# TimeSeries
DF_TS = {}
DF_TS['PC1_hip'] = PC1s[0]
DF_TS['PC1_par'] = PC1s[1]
DF_TS['PC1_pfc'] = PC1s[2]

DF_TS['PC2_hip'] = PC2s[0]
DF_TS['PC2_par'] = PC2s[1]
DF_TS['PC2_pfc'] = PC2s[2]

DF_TS['PC1s_hip'] = PC1s_sm[0]
DF_TS['PC1s_par'] = PC1s_sm[1]
DF_TS['PC1s_pfc'] = PC1s_sm[2]

DF_TS['PC2s_hip'] = PC2s_sm[0]
DF_TS['PC2s_par'] = PC2s_sm[1]
DF_TS['PC2s_pfc'] = PC2s_sm[2]

DF_TS['PC1sh_hip'] = PC1s_shuf[0]
DF_TS['PC1sh_par'] = PC1s_shuf[1]
DF_TS['PC1sh_pfc'] = PC1s_shuf[2]

DF_TS['PC2sh_hip'] = PC2s_shuf[0]
DF_TS['PC2sh_par'] = PC2s_shuf[1]
DF_TS['PC2sh_pfc'] = PC2s_shuf[2]

DF_TS['Expl'] = Exploration
DF_TS['Rodent'] = Rodent
pd.DataFrame(DF_TS).to_pickle(savepath+'TotalSeries.pkl')
