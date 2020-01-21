
def RemovePowerlaw(Data,freq):
    def fc(x, a, b):
        return x**a * b
    Hz1 = np.where(freq>=0.5)[0][0] # only from 1 Hz and larger
    Hz2 = np.where(freq<=200.5)[0][-1] # only from 200 Hz and lower
    frq = freq[Hz1:Hz2]
    data = Data[Hz1:Hz2]
    try:
        popt, pcov = curve_fit(fc, frq, data)
    except:
        popt = [np.nan,np.nan]
    residual = data - fc(frq,popt[0],popt[1])
    return np.array(residual),np.array(frq)

def windowmean(Data,size):
    if size==1: return Data
    elif size==0:
        print('Size 0 not possible!')
        sys.exit()
    else:
        Result=np.zeros(len(Data))+np.nan
        for i in range(np.int(size/2.),np.int(len(Data)-size/2.)):
            Result[i]=np.nanmean(Data[i-np.int(size/2.):i+np.int(size/2.)])
        return np.array(Result)
    
def Abundances(Data,frq):
    borders = np.arange(1,102,1)-0.5
    abund = []
    for i in range(len(borders)-1):
        abund.append(np.nanmean(Data[(frq>borders[i]) & (frq<=borders[i+1])]))
    return abund
        
def MeanCorrelation(Data,ws,N):
    abundsa = []
    means = []
    steps = np.arange(ws,ws+50*N,50)
    sers = Data[:,ws-ws:ws+ws]
    freq_red = rfftfreq(len(sers[0]),d=1e-3)
    for step in tqdm(steps,file=sys.stdout):
        ffts = []
        sers = Data[:,step-ws:step+ws]
        abunds = []
        for i in range(len(sers)):
            FFT = np.abs(rfft(sers[i]))/len(sers[i])*2#,2)
            FFT = FFT[~np.isnan(FFT)]/np.sum(FFT)
            FFTn,FRQn = RemovePowerlaw(FFT,freq_red)
            ffts.append(FFTn)
            abunds.append(Abundances(FFTn,FRQn))
        abundsa.append(np.nanmean(abunds,axis=0))
        means.append(np.nanmean(np.corrcoef(ffts)))
    return np.array(means),np.array(abundsa),np.array(steps)

def Rmetric(array):
    Rmet = np.zeros(shape=(len(array),len(array)))
    for i in range(len(array)):
        for j in range(len(array)):
            Px = np.array(array[i])
            Py = np.array(array[j])
            R = np.nanmean((Px-Py)**2./(Px**2.+Py**2.))
            Rmet[i,j] = 1-R
    return Rmet

def Diffmetric(Rmet):
    Dmet = np.zeros(shape=(len(Rmet),len(Rmet)))
    for i in range(len(Rmet)):
        for j in range(len(Rmet)):
            R = Rmet[i,j]
            D = np.nansum((R-np.mean(Rmet))**2./(np.max(Rmet)-np.mean(Rmet))**2.)
            Dmet[i,j] = D
    return Dmet
        
def MeanRmetric(Data,ws,N):
    abundsa = []
    means = []
    cors = []
    corsdif = []
    diffs = []
    Rmats = []
    Cmats = []
    steps = np.arange(ws,ws+50*N,50)
    sers = Data[:,ws-ws:ws+ws]
    freq_red = rfftfreq(len(sers[0]),d=1e-3)
    for step in tqdm(steps,file=sys.stdout):
        ffts = []
        sers = Data[:,step-ws:step+ws]
        abunds = []
        for i in range(len(sers)):
            FFT = np.abs(rfft(sers[i]))/len(sers[i])*2#,2)
            FFT = FFT[~np.isnan(FFT)]/np.sum(FFT)
            FFTn,FRQn = RemovePowerlaw(FFT,freq_red)
            ffts.append(FFTn)
            abunds.append(Abundances(FFTn,FRQn))
        abundsa.append(np.nanmean(abunds,axis=0))
        Rmet = Rmetric(ffts)
        Cmet = np.corrcoef(ffts)
        means.append(np.nanmean(Rmet))
        diffs.append(np.nanmean(Diffmetric(Rmet)))
        cors.append(np.nanmean(Cmet))
        corsdif.append(np.nanmean(Diffmetric(Cmet)))
        Rmats.append(Rmet)
        Cmats.append(Cmet)
    return np.array(means),np.array(diffs),np.array(cors),np.array(corsdif),np.array(abundsa),np.array(steps),np.array(Cmats)