import numpy as np
import pandas as pd


def func_locations(Str, N, prt, path_pdata):
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

    T1 = 0
    T2 = len(Xpc)-tau

    xe = np.array([xedges2]*len(Ypc[T1:T2]))
    ye = np.array([yedges2]*len(Xpc[T1:T2]))

    Locations = []
    for j in [0]:
        xe_arg = np.abs(xe.T-Ypc[T1+j:T2+j]).argmin(axis=0)
        ye_arg = np.abs(ye.T-Xpc[T1+j:T2+j]).argmin(axis=0)
        Locations.append(ye_arg*len(xedges2)+xe_arg)
    Locations = np.array(Locations)
    return Locations[0]
