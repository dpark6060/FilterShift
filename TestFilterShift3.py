#!/usr/bin/env python

import scipy as sp
import numpy as np,sys
import nibabel as nb
import scipy.signal as sig
import matplotlib.pyplot as pl
from multiprocessing import Pool
import multiprocessing
import time


def MakeT(Int,Start,NumSlices,Tr):
    dt=float(Tr)/float(NumSlices)
    IntSeq=[]
    
    for i in range(Int):
        IntSeq=IntSeq+range(i,NumSlices,Int)
        
    IntSeq=np.array(IntSeq)
    IntSeq=np.roll(IntSeq,Start,0)
    IntTime=IntSeq*dt
    
    return(IntSeq,IntTime)


def bandpass(CutOffFreq, SamplingRate, StopGain, TranWidth):
    NiquistRate = SamplingRate/2.0
    N, beta = sig.kaiserord(StopGain,TranWidth/NiquistRate)
    print 'the order of the FIR filter is:' + str(N) + ', If this is bigger than the size of the data please adjust the width and gain of the filter'
    taps = sig.firwin(N, CutOffFreq, window=('kaiser', beta), pass_zero=False, scale=True, nyq=NiquistRate)

    return (taps,NiquistRate)

#
def loadNIIimg(Fname):
    img=nb.load(Fname).get_data()
    return img



def FiltShift(ZTshift):
    Z=ZTshift[0]
    tShift=ZTshift[1]

    Ny=45

    #print('SmallSig')
    print("STarting Slice "+ str(Z))
    Sig=np.squeeze(Im[:,:,Z,:])
    tdim=len(list(np.shape(Sig)))-1    
    

    LTS=Sig.shape[-1]
    FR=np.array(range(int(round(LTS/2.)),0,-1))
    BR=np.array(range(LTS-1,int(FR.shape[-1])-1,-1))


    if tdim==0:
        
        FrontPad=(Sig[FR])	
        BackPad=Sig[BR]
        LFP=FrontPad.shape[-1]
    elif tdim==1:
        
        FrontPad=Sig[:,FR]
        BackPad=Sig[:,BR]
        

        LFP=FrontPad.shape[-1]

    elif tdim==2:
        
        FrontPad=Sig[:,:,FR]
        BackPad=Sig[:,:,BR]

        LFP=FrontPad.shape[-1]

    elif tdim==3:
        
        FrontPad=Sig[:,:,:,FR]
        BackPad=Sig[:,:,:,BR]

        LFP=FrontPad.shape[-1]

    else:
        print('Bad Array Dimensions for Padding')

    Sig=np.concatenate((FrontPad,Sig,BackPad),-1)
    print('ZeroPadding')

    S=int(round(Fnew/Foriginal))
    Dm=list(np.shape(Sig))
    SS=Dm
    tdim=len(Dm)-1
    Dm[tdim]=Dm[tdim]*S


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
    
    print('Padding Finished')

    del Sig    
    print ZeroSig.shape[-1]
    print('Filtering')
    ZeroSig = sig.filtfilt(BPF, [1], ZeroSig*float(S), axis=-1,padtype='even', padlen=0) 
    print('Filtering Finished')


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
            
        Sig=Sig[LFP:LFP+LTS]
        
    elif tdim==1:  
     
        if shift>0:
            Rep=np.tile(np.expand_dims(ZeroSig[:,0],axis=-1),[1,shift])
            ZeroSig=np.append(Rep,ZeroSig,-1)
            Sig=ZeroSig[:,range(0,ZeroSig.shape[-1]-S,S)]       
    
        else:
            Rep=p.tile(np.expand_dims(ZeroSig,axis=-1),[1,abs(shift)])
            ZeroSig=np.append(ZeroSig,Rep,-1)
            Sig=ZeroSig[:,range(int(abs(shift)),ZeroSig.shape[-1]-1,S)]
            
        Sig=Sig[:,LFP:LFP+LTS]
  
    elif tdim==2: 

        if shift>0:
    
            Rep=np.tile(np.expand_dims(ZeroSig[:,:,0],axis=-1),[1,1,shift])
            ZeroSig=np.append(Rep,ZeroSig,-1)
            Sig=ZeroSig[:,:,range(0,ZeroSig.shape[-1]-S,S)]       
    
        else:
            Rep=np.tile(np.expand_dims(ZeroSig[:,:,-1],axis=-1),[1,1,abs(shift)])
            ZeroSig=np.append(ZeroSig,Rep,-1)
            Sig=ZeroSig[:,:,range(int(abs(shift)),ZeroSig.shape[-1]-1,S)]
        
        Sig=Sig[:,:,LFP:LFP+LTS]

       
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
        
        Sig=Sig[:,:,:,LFP:LFP+LTS]

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
SliceOrder, SliceShift = MakeT(Int, 0, Nz, Tr)
Slices = np.array(range(len(SliceShift)))
inputs = np.column_stack((Slices, SliceShift))  


p=Pool(14)
print("Starting Pool")
print("Pool Started, Running at " + str(time.ctime(time.clock())))
FiltSigs = p.map(FiltShift, inputs)

FnameTwo = "/home/dparker/Desktop/Sub4Processed2.nii"
FiltSigs=np.array(FiltSigs)
FiltSigs=np.swapaxes(np.swapaxes(FiltSigs,0,1),1,2)
temp=nb.load(Fname)
img=nb.Nifti1Image(FiltSigs,np.eye(4))

nb.loadsave.save(img,FnameTwo)

img=nb.Nifti1Image(FiltSigs,temp.get_qform(),temp.get_header())

nb.loadsave.save(img,FnameTwo)



