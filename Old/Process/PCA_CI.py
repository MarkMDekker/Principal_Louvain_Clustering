# --------------------------------------------------------------------- #
# Preambule
# --------------------------------------------------------------------- #

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.io
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
    # Get overall EOFs
    # --------------------------------------------------------------------- #
    
    EOFs = np.array(pd.read_pickle(path_pdata+'EOFs/'+Str+'/600000_0.pkl'))
    
    # --------------------------------------------------------------------- #
    # Read data
    # --------------------------------------------------------------------- #
    
    ProcessedPower = np.array(pd.read_pickle(path_pdata+'ProcessedPower/' +
                                             s+'.pkl'))
    PCs = []
    for i in range(len(ProcessedPower)):
        PCs.append(EOFs[i].dot(ProcessedPower))
    
    # --------------------------------------------------------------------- #
    # Save data
    # --------------------------------------------------------------------- #
    
    pd.DataFrame(PCs).to_pickle(path_pdata+'PCs/'+s+'.pkl')
