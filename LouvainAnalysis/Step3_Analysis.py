# ----------------------------------------------------------------- #
# About script
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import configparser
import pandas as pd
import scipy.stats
from tqdm import tqdm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
figpath = ('/Users/mmdekker/Documents/Werk/Figures/Figures_Side/Brain/'
           'Pics_20201020/')
savepath = '/Users/mmdekker/Documents/Werk/Data/Neuro/Processed/'

# ----------------------------------------------------------------- #
# Input
# ----------------------------------------------------------------- #

RES = 10
TAU = 30
SUBs = ['279418', '279419', '339295', '339296', '340233', '341319',
        '341321', '346110', '346111', '347640', '347641']
space = np.linspace(-12, 12, RES)

# ----------------------------------------------------------------- #
# Read data
# ----------------------------------------------------------------- #

Series = pd.read_pickle(savepath+'TotalSeries.pkl')
#%%

SE = np.array(Series).T
PC1_hip, PC1_par, PC1_pfc, PC2_hip, PC2_par, PC2_pfc = SE[:6]
PC1s_hip, PC1s_par, PC1s_pfc, PC2s_hip, PC2s_par, PC2s_pfc = SE[6:12]
PC1sh_hip, PC1sh_par, PC1sh_pfc, PC2sh_hip, PC2sh_par, PC2sh_pfc = SE[12:18]
Expl, ROD, cell, cluster, cell_s, cluster_s, cell_sh, cluster_sh = SE[18:]
AvExpl = len(Expl[Expl > 1]) / len(Expl[Expl >= 1])
PC1_hip, PC1_par, PC1_pfc, PC2_hip, PC2_par, PC2_pfc, PC1s_hip, PC1s_par, PC1s_pfc, PC2s_hip, PC2s_par, PC2s_pfc, PC1sh_hip, PC1sh_par, PC1sh_pfc, PC2sh_hip, PC2sh_par, PC2sh_pfc = np.array([PC1_hip, PC1_par, PC1_pfc, PC2_hip, PC2_par, PC2_pfc, PC1s_hip, PC1s_par, PC1s_pfc, PC2s_hip, PC2s_par, PC2s_pfc, PC1sh_hip, PC1sh_par, PC1sh_pfc, PC2sh_hip, PC2sh_par, PC2sh_pfc]).astype(float)
pd.DataFrame(Expl).to_pickle(savepath+'Expl.pkl')
pd.DataFrame(cluster_s).to_pickle(savepath+'Clus_s.pkl')
pd.DataFrame(cluster_sh).to_pickle(savepath+'Clus_sh.pkl')
#%%
# ----------------------------------------------------------------- #
# Create meta DF and amount data / unique cells
# ----------------------------------------------------------------- #

DF_meta = {}
DF_meta['Clusters'] = np.unique(cluster)
n = []
c = []
for i in np.unique(cluster):
    wh = np.where(cluster == i)[0]
    n.append(len(wh))
    c.append(len(np.unique(cell[wh])))
DF_meta['AmountData'] = n
DF_meta['UniqueCells'] = c
DF_meta = pd.DataFrame(DF_meta)

DF_meta_s = {}
DF_meta_s['Clusters'] = np.unique(cluster_s)
n = []
c = []
for i in np.unique(cluster_s):
    wh = np.where(cluster_s == i)[0]
    n.append(len(wh))
    c.append(len(np.unique(cell_s[wh])))
DF_meta_s['AmountData'] = n
DF_meta_s['UniqueCells'] = c
DF_meta_s = pd.DataFrame(DF_meta_s)

DF_meta_sh = {}
DF_meta_sh['Clusters'] = np.unique(cluster_sh)
n = []
c = []
for i in np.unique(cluster_sh):
    wh = np.where(cluster_sh == i)[0]
    n.append(len(wh))
    c.append(len(np.unique(cell_sh[wh])))
DF_meta_sh['AmountData'] = n
DF_meta_sh['UniqueCells'] = c
DF_meta_sh = pd.DataFrame(DF_meta_sh)


# ----------------------------------------------------------------- #
# Column: bias per rodent
# ----------------------------------------------------------------- #

explbiases = np.zeros(shape=(len(SUBs), len(DF_meta.Clusters)))
explbiasav = np.zeros(len(DF_meta.Clusters))
for i in tqdm(range(len(SUBs))):
    dat = Expl[(ROD == i) & (Expl >= 1)]
    av_expl = len(np.where(dat >= 2)[0])/len(dat)
    for c in range(len(DF_meta.Clusters)):
        datc = Expl[(ROD == i) & (cluster == c) & (Expl >= 1)]
        if len(datc) > 0:
            c_expl = len(np.where(datc >= 2)[0])/len(datc)
        else:
            c_expl = 0
        explbiases[i, c] = c_expl / AvExpl
        if i == 0:
            explbiasav[c] = len(np.where(Expl[cluster == c] >= 2)[0])/len(Expl[cluster == c])/AvExpl

DF_meta['Explbias'] = explbiasav
DF_meta['Absbias'] = np.abs(1-explbiasav)
for i in range(11):
    DF_meta['Explbias_r'+str(i+1)] = explbiases[i]
#DF_meta['Explbias_r2'] = explbiases[1]
#DF_meta['Explbias_r3'] = explbiases[2]
#DF_meta['Explbias_r4'] = explbiases[3]
#DF_meta['Explbias_r5'] = explbiases[4]
#DF_meta['Explbias_r6'] = explbiases[5]

explbiases_s = np.zeros(shape=(len(SUBs), len(DF_meta_s.Clusters)))
explbiasav_s = np.zeros(len(DF_meta_s.Clusters))
for i in tqdm(range(len(SUBs))):
    dat = Expl[(ROD == i) & (Expl >= 1)]
    av_expl = len(np.where(dat >= 2)[0])/len(dat)
    for c in range(len(DF_meta_s.Clusters)):
        datc = Expl[(ROD == i) & (cluster_s == c) & (Expl >= 1)]
        if len(datc) > 0:
            c_expl = len(np.where(datc >= 2)[0])/len(datc)
        else:
            c_expl = 0
        explbiases_s[i, c] = c_expl / AvExpl
        if i == 0:
            explbiasav_s[c] = len(np.where(Expl[cluster_s == c] >= 2)[0])/len(Expl[cluster_s == c])/AvExpl

DF_meta_s['Explbias'] = explbiasav_s
DF_meta_s['Absbias'] = np.abs(1-explbiasav_s)
for i in range(11):
    DF_meta_s['Explbias_r'+str(i+1)] = explbiases_s[i]
# DF_meta_s['Explbias_r1'] = explbiases_s[0]
# DF_meta_s['Explbias_r2'] = explbiases_s[1]
# DF_meta_s['Explbias_r3'] = explbiases_s[2]
#DF_meta_s['Explbias_r4'] = explbiases_s[3]
#DF_meta_s['Explbias_r5'] = explbiases_s[4]
#DF_meta_s['Explbias_r6'] = explbiases_s[5]
#%%
explbiases_sh = np.zeros(shape=(len(SUBs), len(DF_meta_sh.Clusters)))
explbiasav_sh = np.zeros(len(DF_meta_sh.Clusters))
for i in tqdm(range(len(SUBs))):
    dat = Expl[(ROD == i) & (Expl >= 1)]
    av_expl = len(np.where(dat >= 2)[0])/len(dat)
    for c in range(len(DF_meta_sh.Clusters)):
        datc = Expl[(ROD == i) & (cluster_sh == c) & (Expl >= 1)]
        if len(datc) > 0:
            c_expl = len(np.where(datc >= 2)[0])/len(datc)
        else:
            c_expl = 0
        explbiases_sh[i, c] = c_expl / AvExpl
        if i == 0:
            explbiasav_sh[c] = len(np.where(Expl[cluster_sh == c] >= 2)[0])/len(Expl[cluster_sh == c])/AvExpl

DF_meta_sh['Explbias'] = explbiasav_sh
DF_meta_sh['Absbias'] = np.abs(1-explbiasav_sh)
for i in range(11):
    DF_meta_sh['Explbias_r'+str(i+1)] = explbiases_sh[i]
# DF_meta_sh['Explbias_r1'] = explbiases_sh[0]
# DF_meta_sh['Explbias_r2'] = explbiases_sh[1]
# DF_meta_sh['Explbias_r3'] = explbiases_sh[2]
#DF_meta_sh['Explbias_r4'] = explbiases_sh[3]
#DF_meta_sh['Explbias_r5'] = explbiases_sh[4]
#DF_meta_sh['Explbias_r6'] = explbiases_sh[5]

# ----------------------------------------------------------------- #
# Column: % in rodent
# ----------------------------------------------------------------- #

locclus = cluster
Rods = np.zeros(shape=(len(SUBs), len(DF_meta)))
for c in range(len(DF_meta)):
    tot = len(locclus[locclus == c])
    for i in range(len(SUBs)):
        rodtot = len(Expl[(ROD == i) & (locclus == c)])
        Rods[i, c] = rodtot/tot

for i in range(11):
    DF_meta['Perc_r'+str(i+1)] = Rods[i]
#DF_meta['Perc_r2'] = Rods[1]
#DF_meta['Perc_r3'] = Rods[2]
#DF_meta['Perc_r4'] = Rods[3]
#DF_meta['Perc_r5'] = Rods[4]
#DF_meta['Perc_r6'] = Rods[5]

locclus = cluster_s
Rods_s = np.zeros(shape=(len(SUBs), len(DF_meta_s)))
for c in range(len(DF_meta_s)):
    tot = len(locclus[locclus == c])
    for i in range(len(SUBs)):
        rodtot = len(Expl[(ROD == i) & (locclus == c)])
        Rods_s[i, c] = rodtot/tot

for i in range(11):
    DF_meta_s['Perc_r'+str(i+1)] = Rods_s[i]
# DF_meta_s['Perc_r1'] = Rods_s[0]
# DF_meta_s['Perc_r2'] = Rods_s[1]
# DF_meta_s['Perc_r3'] = Rods_s[2]
#DF_meta_s['Perc_r4'] = Rods_s[3]
#DF_meta_s['Perc_r5'] = Rods_s[4]
#DF_meta_s['Perc_r6'] = Rods_s[5]

locclus = cluster_sh
Rods_sh = np.zeros(shape=(len(SUBs), len(DF_meta_sh)))
for c in range(len(DF_meta_sh)):
    tot = len(locclus[locclus == c])
    for i in range(len(SUBs)):
        rodtot = len(Expl[(ROD == i) & (locclus == c)])
        Rods_sh[i, c] = rodtot/tot

for i in range(11):
    DF_meta_sh['Perc_r'+str(i+1)] = Rods_sh[i]
# DF_meta_sh['Perc_r1'] = Rods_sh[0]
# DF_meta_sh['Perc_r2'] = Rods_sh[1]
# DF_meta_sh['Perc_r3'] = Rods_sh[2]
#DF_meta_sh['Perc_r4'] = Rods_sh[3]
#DF_meta_sh['Perc_r5'] = Rods_sh[4]
#DF_meta_sh['Perc_r6'] = Rods_sh[5]
#%%
# ----------------------------------------------------------------- #
# Column: region-biases
# ----------------------------------------------------------------- #

varhip0 = np.nanmean([np.nanvar(PC1_hip), np.nanvar(PC2_hip)])
varpar0 = np.nanmean([np.nanvar(PC1_par), np.nanvar(PC2_par)])
varpfc0 = np.nanmean([np.nanvar(PC1_pfc), np.nanvar(PC2_pfc)])

varsy = np.zeros(shape=(3, len(DF_meta)))
for c in tqdm(range(len(DF_meta))):
    VarHIP = np.nanmean([np.nanvar(PC1_hip[cluster == c]), np.nanvar(PC2_hip[cluster == c])])/varhip0
    VarPAR = np.nanmean([np.nanvar(PC1_par[cluster == c]), np.nanvar(PC2_par[cluster == c])])/varpar0
    VarPFC = np.nanmean([np.nanvar(PC1_pfc[cluster == c]), np.nanvar(PC2_pfc[cluster == c])])/varpfc0
    varsy[0, c] = VarHIP
    varsy[1, c] = VarPAR
    varsy[2, c] = VarPFC

DF_meta['VarHIP'] = varsy[0]
DF_meta['VarPAR'] = varsy[1]
DF_meta['VarPFC'] = varsy[2]
DF_meta['InterReg'] = np.nanmax(varsy/np.nansum(varsy, axis=0), axis=0)

varhip0 = np.nanmean([np.nanvar(PC1s_hip), np.nanvar(PC2s_hip)])
varpar0 = np.nanmean([np.nanvar(PC1s_par), np.nanvar(PC2s_par)])
varpfc0 = np.nanmean([np.nanvar(PC1s_pfc), np.nanvar(PC2s_pfc)])

varsys = np.zeros(shape=(3, len(DF_meta_s)))
for c in tqdm(range(len(DF_meta_s))):
    VarHIP = np.nanmean([np.nanvar(PC1s_hip[cluster_s == c]), np.nanvar(PC2s_hip[cluster_s == c])])/varhip0
    VarPAR = np.nanmean([np.nanvar(PC1s_par[cluster_s == c]), np.nanvar(PC2s_par[cluster_s == c])])/varpar0
    VarPFC = np.nanmean([np.nanvar(PC1s_pfc[cluster_s == c]), np.nanvar(PC2s_pfc[cluster_s == c])])/varpfc0
    varsys[0, c] = VarHIP
    varsys[1, c] = VarPAR
    varsys[2, c] = VarPFC

DF_meta_s['VarHIP'] = varsys[0]
DF_meta_s['VarPAR'] = varsys[1]
DF_meta_s['VarPFC'] = varsys[2]
DF_meta_s['InterReg'] = np.nanmax(varsys/np.nansum(varsys, axis=0), axis=0)

varhip0 = np.nanmean([np.nanvar(PC1sh_hip), np.nanvar(PC2sh_hip)])
varpar0 = np.nanmean([np.nanvar(PC1sh_par), np.nanvar(PC2sh_par)])
varpfc0 = np.nanmean([np.nanvar(PC1sh_pfc), np.nanvar(PC2sh_pfc)])

varsys = np.zeros(shape=(3, len(DF_meta_sh)))
for c in tqdm(range(len(DF_meta_sh))):
    VarHIP = np.nanmean([np.nanvar(PC1sh_hip[cluster_sh == c]), np.nanvar(PC2sh_hip[cluster_sh == c])])/varhip0
    VarPAR = np.nanmean([np.nanvar(PC1sh_par[cluster_sh == c]), np.nanvar(PC2sh_par[cluster_sh == c])])/varpar0
    VarPFC = np.nanmean([np.nanvar(PC1sh_pfc[cluster_sh == c]), np.nanvar(PC2sh_pfc[cluster_sh == c])])/varpfc0
    varsys[0, c] = VarHIP
    varsys[1, c] = VarPAR
    varsys[2, c] = VarPFC

DF_meta_sh['VarHIP'] = varsys[0]
DF_meta_sh['VarPAR'] = varsys[1]
DF_meta_sh['VarPFC'] = varsys[2]
DF_meta_sh['InterReg'] = np.nanmax(varsys/np.nansum(varsys, axis=0), axis=0)

# ----------------------------------------------------------------- #
# Column: average residence time
# ----------------------------------------------------------------- #

avres = []
incid = []
for c in range(len(DF_meta)):
    wh = np.where(cluster == c)[0]
    ts = []
    t = 1
    for i in range(1, len(wh)):
        if wh[i] <= wh[i-1]+30:
            t += 1
        else:
            ts.append(t)
            t = 1
    if t != 0:
        try:
            if ts[-1] != t:
                ts.append(t)
        except:
            if len(ts) == 0:
                ts.append(t)
    ts = np.array(ts)
    avres.append(np.mean(ts[ts > 2]))
    incid.append(len(ts[ts > 2]))
DF_meta['AvResTime'] = avres
DF_meta['UniquePresences'] = incid

avres_s = []
incid_s = []
for c in range(len(DF_meta_s)):
    wh = np.where(cluster_s == c)[0]
    ts = []
    t = 1
    for i in range(1, len(wh)):
        if wh[i] <= wh[i-1]+30:
            t += 1
        else:
            ts.append(t)
            t = 1
    if t != 0:
        try:
            if ts[-1] != t:
                ts.append(t)
        except:
            if len(ts) == 0:
                ts.append(t)
    ts = np.array(ts)
    avres_s.append(np.mean(ts[ts > 2]))
    incid_s.append(len(ts[ts > 2]))
DF_meta_s['AvResTime'] = avres_s
DF_meta_s['UniquePresences'] = incid_s

avres_sh = []
incid_sh = []
for c in range(len(DF_meta_sh)):
    wh = np.where(cluster_sh == c)[0]
    ts = []
    t = 1
    for i in range(1, len(wh)):
        if wh[i] <= wh[i-1]+30:
            t += 1
        else:
            ts.append(t)
            t = 1
    if t != 0:
        try:
            if ts[-1] != t:
                ts.append(t)
        except:
            if len(ts) == 0:
                ts.append(t)
    ts = np.array(ts)
    avres_sh.append(np.mean(ts[ts > 2]))
    incid_sh.append(len(ts[ts > 2]))
DF_meta_sh['AvResTime'] = avres_sh
DF_meta_sh['UniquePresences'] = incid_sh

# ----------------------------------------------------------------- #
# Resave
# ----------------------------------------------------------------- #

DF_meta.to_csv(savepath+'ClusterAnalysis.csv')
DF_meta_s.to_csv(savepath+'ClusterAnalysis_s.csv')
DF_meta_sh.to_csv(savepath+'ClusterAnalysis_sh.csv')
