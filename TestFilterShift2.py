#!/usr/bin/env python

import scipy as sp
import numpy as np,sys
import nibabel as nb
import scipy.signal as sig
import matplotlib.pyplot as pl
from multiprocessing import Pool
import multiprocessing
import time
import gc
import MakeIntTime

SliceShift=np.array([])

def bandpass(CutOffFreq, SamplingRate, StopGain, TranWidth):
    NiquistRate = SamplingRate/2.0
    N, beta = sig.kaiserord(StopGain,TranWidth/NiquistRate)
    print 'the order of the FIR filter is:' + str(N) + ', If this is bigger than the size of the data please adjust the width and gain of the filter'
    taps = sig.firwin(N, CutOffFreq, window=('kaiser', beta), pass_zero=False, scale=True, nyq=NiquistRate)

    return (taps,NiquistRate)


def loadNIIimg(Fname):
    img=nb.load(Fname).get_data()
    return img




def FiltShift(ZTshift):
    Z=ZTshift
    tShift=SliceShift[Z]

    Ny=45

    #print('SmallSig')


    print("STarting Slice "+ str(Z))
    Sig=np.squeeze(Im[:,Ny,Z,:])
    
    S=int(round(Fnew/Foriginal))
    Dm=list(np.shape(Sig))
    SS=Dm
    tdim=len(Dm)-1
    Dm[tdim]=Dm[tdim]*S

    #print('ZeroPadding')
    ZeroSig=np.zeros(Dm)

    k=0
    if tdim==0:
        ZeroSig[::S]=Sig
    elif tdim==1:       
        ZeroSig[:,::S]=Sig
    elif tdim==2:        
        ZeroSig[:,:,::S]=Sig                 
    elif tdim==3:       
        ZeroSig[:,:,:,::S]=Sig
    else:
        print("Bad Array Dimensions")  
    
    #print('Padding Finished')

    del Sig    

    LenOfTimeSeries=ZeroSig.shape[-1]
    #print('Filtering')
    ZeroSig = sig.filtfilt(BPF, [1], ZeroSig*float(S), axis=-1,padtype='even', padlen=LenOfTimeSeries/2) 
    #print('Filtering Finished')


    #print('Shifting')

    tdim=len(SS)-1
    Fs=Fnew
    Ts=1/Fs

    Sig=np.zeros(SS)
    shift=round(tShift*Fs)
    print shift

    if tdim==0:
    
        if shift>0:
            print ZeroSig.shape
            Rep=np.tile(ZeroSig[0],shift)
            ZeroSig=np.append(Rep,ZeroSig,-1)
            print ZeroSig.shape
            Sig=ZeroSig[range(0,ZeroSig.shape[-1]-S,S)]       
        
        else:
            Rep=np.tile(ZeroSig[-1],abs(shift))
            ZeroSig=np.append(ZeroSig,Rep,-1)
            Sig=ZeroSig[range(int(abs(shift)),ZeroSig.shape[-1]-1,S)]       
         
    elif tdim==1:  
     
        if shift>0:
            Rep=np.tile(np.expand_dims(ZeroSig[:,0],axis=-1),[1,shift])
            ZeroSig=np.append(Rep,ZeroSig,-1)
            Sig=ZeroSig[:,range(0,ZeroSig.shape[-1]-S,S)]       
    
        else:
            Rep=p.tile(np.expand_dims(ZeroSig,axis=-1),[1,abs(shift)])
            ZeroSig=np.append(ZeroSig,Rep,-1)
            Sig=ZeroSig[:,range(int(abs(shift)),ZeroSig.shape[-1]-1,S)]     
    
  
    elif tdim==2: 

        if shift>0:
    
            Rep=np.tile(np.expand_dims(ZeroSig[:,:,0],axis=-1),[1,1,shift])
            ZeroSig=np.append(Rep,ZeroSig,-1)
            Sig=ZeroSig[:,:,range(0,ZeroSig.shape[-1]-S,S)]       
    
        else:
            Rep=np.tile(np.expand_dims(ZeroSig[:,:,-1],axis=-1),[1,1,abs(shift)])
            ZeroSig=np.append(ZeroSig,Rep,-1)
            Sig=ZeroSig[:,:,range(int(abs(shift)),ZeroSig.shape[-1]-1,S)]     

       
    elif tdim==3:   
        if shift>0:
    
            Rep=np.tile(np.expand_dims(ZeroSig[:,:,:,0],axis=-1),[1,1,1,shift])
            Rep=np.expand_dims(Rep,axis=-1)
            ZeroSig=np.append(Rep,ZeroSig,-1)
            Sig=ZeroSig[:,:,:,range(0,ZeroSig.shape[-1]-S,S)]       
    
        else:
            Rep=np.tile(np.expand_dims(ZeroSig[:,:,:,-1],axis=-1),[1,1,1,shift])

            Rep=np.expand_dims(Rep,axis=-1)
            ZeroSig=np.append(ZeroSig,Rep,-1)
            Sig=ZeroSig[:,:,:,range(int(abs(shift)),ZeroSig.shape[-1]-1,S)]     

    else:
        
            print("Bad Array Dimensions")  
    

    print("Finished Slice "+str(Z))

    print Sig.shape
#        out_q.put(Sig)
    return Sig
   
       

Fname = "/Data/DavidSub_4/SimulatedBoldSmooth_66_Noise0_Interleaved6.nii.gz"
    
Im = loadNIIimg(Fname)
print("Image Loaded")   
Lt = Im.shape[3]
Nz = Im.shape[2]
Ny = Im.shape[1]
Nx = Im.shape[0]
Tr = 2.0
Int = 6
Foriginal = 0.5 # Hz
Fnew = 20.0 #Hz
Stopgain = 60
Tranwidth = 0.08
print("Designing Filter")    
BPF, Nyq = bandpass([0.01, 0.21], Fnew, Stopgain, Tranwidth)
print("Done")

SliceOrder, SliceShift = MakeIntTime.MakeT(Int, 0, Nz, Tr)  


p=Pool(2)
print("Starting Pool")
print("Pool Started, Running at " + str(time.ctime(time.clock())))
inputs=range(3)
FiltSigs = p.map(FiltShift, inputs)
p.close()
p.join()

