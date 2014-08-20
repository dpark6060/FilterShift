import numpy as np
import sys
import ZeroNorm
import matplotlib.pyplot as pl

def MakeFig():
	f=pl.figure()
	ax1=f.add_subplot(211)
	ax2=f.add_subplot(212)
	return(f,ax1,ax2)

def AddData(dax,fax,Data,Fs,label):

    N=Data.shape[0]
    t=np.arange(0,N/float(Fs),1/Fs)	
    #dax.plot(t,ZeroNorm.Normalize(ZeroNorm.Zero(Data)),label=label)
    dax.plot(t,Data,label=label)
    #F=np.arange(-Fs/2.,Fs/2,Fs/float(N))    
    F=np.linspace(-Fs/2,Fs/2,N)	

    fData=np.fft.fftshift(np.fft.fft(Data))
    fData[round(len(fData)/2)]=1
    fax.plot(F,np.abs(fData),label=label)
    fax.set_yscale('log')

    return (dax,fax)

def ShowPlot():


    pl.xlabel('Samples')
    pl.ylabel('magnitude')
    pl.legend()
    pl.show()
    
