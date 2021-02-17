# --------------------------------------------------------------------- #
# Preambule
# --------------------------------------------------------------------- #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
path_pdata = ('/Users/mmdekker/Documents/Werk/Data/SideProjects/Braindata/'
              'ProcessedData/')

# --------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------- #

Str = 'PAR'
N = 100000
prt = 0
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
T2 = len(Xpc)

Rates = np.zeros(shape=(res2**2,res2**2))
DFindex = np.array(DF.index)
DFxi = np.array(DF.xi)
DFyi = np.array(DF.yi)
xe = np.array([xedges2]*len(Ypc))
ye = np.array([yedges2]*len(Xpc))

Locations = []
for j in [0]:
    xe_arg = np.abs(xe.T-Ypc[T1+j:T2+j]).argmin(axis=0)
    ye_arg = np.abs(ye.T-Xpc[T1+j:T2+j]).argmin(axis=0)
    Locations.append(ye_arg*len(xedges2)+xe_arg)
Locations = np.array(Locations)
unilocations = np.unique(Locations[0])

#%%
# --------------------------------------------------------------------- #
# Exismatrix
# --------------------------------------------------------------------- #

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

ExisMat = np.zeros(len(xe))
ExisMat[unilocations] = 1
ExisMat = func12(ExisMat)

# --------------------------------------------------------------------- #
# 2D ratesmat
# --------------------------------------------------------------------- #



#%%
# --------------------------------------------------------------------- #
# Get overall EOFs
# --------------------------------------------------------------------- #

EOFs = np.array(pd.read_pickle(path_pdata+'EOFs/HR_'+Str+'.pkl'))

# --------------------------------------------------------------------- #
# Read data
# --------------------------------------------------------------------- #

ProcessedPower = np.array(pd.read_pickle(path_pdata+'ProcessedPower/' +
                                         s+'.pkl'))
PCs = []
for i in range(len(ProcessedPower)-1):
    PCs.append(EOFs[i].dot(ProcessedPower))

# --------------------------------------------------------------------- #
# Save data
# --------------------------------------------------------------------- #

pd.DataFrame(PCs).to_pickle(path_pdata+'PCs/'+s+'.pkl')
