#!/usr/bin/env python

import scipy as sp
import numpy as np,sys
import nibabel as nb
import scipy.signal as sig
from scipy.interpolate import interp1d
import matplotlib.pyplot as pl
from multiprocessing import Pool
import multiprocessing
import time
import MakeIntTime
import ZeroNorm

def Sinc(Data,Fs,Tshift):
    
    
    if Tshift!=0:
        tdim=Data.ndim-1
        
        Shift=Tshift*Fs
        pad=int(np.ceil(Shift))   
    
    
        if tdim == 0:
        
            PF=Data[0]
            PB=Data[-1]            
            PF=np.tile(PF,pad)
            PB=np.tile(PB,pad)
            
            Data=np.concatenate((PF,Data,PB),-1)
            n=Data.shape
            lnth=n[-1]
            F=np.resize(np.linspace(-Fs/2.,Fs/2.,lnth),Data.shape)                     
            Fsig=np.fft.fftshift(np.fft.fft(Data))
            Fshift=Fsig*np.exp(-1j*Shift*F*2.*np.pi/Fs)
            Data=np.real(np.fft.ifft(np.fft.ifftshift(Fshift)))
            Data=Data[pad:-pad]
            
        elif tdim == 1:
            PF=Data[:,0]
            PB=Data[:,-1]            
            PF=np.tile(np.expand_dims(PF,-1),[1,pad])
            PB=np.tile(np.expand_dims(PB,-1),[1,pad])
            
            Data=np.concatenate((PF,Data,PB),-1)
            n=Data.shape
            lnth=n[-1]
            F=np.resize(np.linspace(-Fs/2.,Fs/2.,lnth),Data.shape)                     
            Fsig=np.fft.fftshift(np.fft.fft(Data))
            Fshift=Fsig*np.exp(-1j*Shift*F*2.*np.pi/Fs)
            Data=np.real(np.fft.ifft(np.fft.ifftshift(Fshift)))
            
            Data = Data[:,pad:-pad]
            
        elif tdim == 2:
            PF=Data[:,:,0]
            PB=Data[:,:,-1]            
            PF=np.tile(np.expand_dims(PF,-1),[1,1,pad])
            PB=np.tile(np.expand_dims(PB,-1),[1,1,pad])
            
            Data=np.concatenate((PF,Data,PB),-1)
            n=Data.shape
            lnth=n[-1]
            F=np.resize(np.linspace(-Fs/2.,Fs/2.,lnth),Data.shape)                     
            Fsig=np.fft.fftshift(np.fft.fft(Data))
            Fshift=Fsig*np.exp(-1j*Shift*F*2.*np.pi/Fs)
            Data=np.real(np.fft.ifft(np.fft.ifftshift(Fshift)))
            
            Data = Data[:,:,pad:-pad]
            
        elif tdim == 3:
            PF=Data[:,:,:,0]
            PB=Data[:,:,:,-1]            
            PF=np.tile(np.expand_dims(PF,-1),[1,1,1,pad])
            PB=np.tile(np.expand_dims(PB,-1),[1,1,1,pad])
            
            Data=np.concatenate((PF,Data,PB),-1)
            n=Data.shape
            lnth=n[-1]
            F=np.resize(np.linspace(-Fs/2.,Fs/2.,lnth),Data.shape)                     
            Fsig=np.fft.fftshift(np.fft.fft(Data))
            Fshift=Fsig*np.exp(-1j*Shift*F*2.*np.pi/Fs)
            Data=np.real(np.fft.ifft(np.fft.ifftshift(Fshift)))
            
            Data = Data[:,:,:,pad:-pad]
        else:
            print('Error In Sinc Interp: Bad Dimensions')
            
    return Data
  
 
def Sinc2(In):
    
    Data=np.squeeze(Im[:,:,In[0],:])
    Fs=In[1]
    Tshift=In[2]
    
    if Tshift!=0:
        tdim=Data.ndim-1
        
        Shift=Tshift*Fs
        pad=int(np.ceil(Shift))   
    
    
        if tdim == 0:
        
            PF=Data[0]
            PB=Data[-1]            
            PF=np.tile(PF,pad)
            PB=np.tile(PB,pad)
            
            Data=np.concatenate((PF,Data,PB),-1)
            n=Data.shape
            lnth=n[-1]
            F=np.resize(np.linspace(-Fs/2.,Fs/2.,lnth),Data.shape)                     
            Fsig=np.fft.fftshift(np.fft.fft(Data))
            Fshift=Fsig*np.exp(-1j*Shift*F*2.*np.pi/Fs)
            Data=np.real(np.fft.ifft(np.fft.ifftshift(Fshift)))
            Data=Data[pad:-pad]
            
        elif tdim == 1:
            PF=Data[:,0]
            PB=Data[:,-1]            
            PF=np.tile(np.expand_dims(PF,-1),[1,pad])
            PB=np.tile(np.expand_dims(PB,-1),[1,pad])
            
            Data=np.concatenate((PF,Data,PB),-1)
            n=Data.shape
            lnth=n[-1]
            F=np.resize(np.linspace(-Fs/2.,Fs/2.,lnth),Data.shape)                     
            Fsig=np.fft.fftshift(np.fft.fft(Data))
            Fshift=Fsig*np.exp(-1j*Shift*F*2.*np.pi/Fs)
            Data=np.real(np.fft.ifft(np.fft.ifftshift(Fshift)))
            
            Data = Data[:,pad:-pad]
            
        elif tdim == 2:
            PF=Data[:,:,0]
            PB=Data[:,:,-1]            
            PF=np.tile(np.expand_dims(PF,-1),[1,1,pad])
            PB=np.tile(np.expand_dims(PB,-1),[1,1,pad])
            
            Data=np.concatenate((PF,Data,PB),-1)
            n=Data.shape
            lnth=n[-1]
            F=np.resize(np.linspace(-Fs/2.,Fs/2.,lnth),Data.shape)                     
            Fsig=np.fft.fftshift(np.fft.fft(Data))
            Fshift=Fsig*np.exp(-1j*Shift*F*2.*np.pi/Fs)
            Data=np.real(np.fft.ifft(np.fft.ifftshift(Fshift)))
            
            Data = Data[:,:,pad:-pad]
            
        elif tdim == 3:
            PF=Data[:,:,:,0]
            PB=Data[:,:,:,-1]            
            PF=np.tile(np.expand_dims(PF,-1),[1,1,1,pad])
            PB=np.tile(np.expand_dims(PB,-1),[1,1,1,pad])
            
            Data=np.concatenate((PF,Data,PB),-1)
            n=Data.shape
            lnth=n[-1]
            F=np.resize(np.linspace(-Fs/2.,Fs/2.,lnth),Data.shape)                     
            Fsig=np.fft.fftshift(np.fft.fft(Data))
            Fshift=Fsig*np.exp(-1j*Shift*F*2.*np.pi/Fs)
            Data=np.real(np.fft.ifft(np.fft.ifftshift(Fshift)))
            
            Data = Data[:,:,:,pad:-pad]
        else:
            print('Error In Sinc Interp: Bad Dimensions')
            
    return Data


def linear2(In):
    
    
    Data=ZeroNorm.Zero(np.squeeze(Im[:,:,In[0],:]))
    Fs=In[1]
    Tshift=In[2]
        
    tdim=Data.ndim-1
    n=Data.shape[-1]
    Tgrid=np.array(range(n))/Fs
    
    f=interp1d(Tgrid,Data,kind='linear',axis=-1,copy=False,bounds_error=False,fill_value=0)
    Data=f(Tgrid-Tshift)    
    
    return Data



def cubic2(In):
    
    Data=ZeroNorm.Zero(np.squeeze(Im[:,:,In[0],:]))
    Fs=In[1]
    Tshift=In[2]
    
    tdim=Data.ndim-1
    n=Data.shape[-1]
    Tgrid=np.array(range(n))/Fs
    
    f=interp1d(Tgrid,Data,kind='cubic',axis=-1,copy=False,bounds_error=False,fill_value=0)
    Data=f(Tgrid-Tshift)
    return Data


def spline2(In):
    
    Data=ZeroNorm.Zero(np.squeeze(Im[:,:,In[0],:]))
    Fs=In[1]
    Tshift=In[2]
    
    tdim=Data.ndim-1
    n=Data.shape[-1]
    Tgrid=np.array(range(n))/Fs
    
    f=interp1d(Tgrid,Data,kind='quadratic',axis=-1,copy=False,bounds_error=False,fill_value=0)
    Data=f(Tgrid-Tshift)
    return Data


def linear(Data,Fs,Tshift):
    
    tdim=Data.ndim-1
    n=Data.shape[-1]
    Tgrid=np.array(range(n))/Fs
    
    f=interp1d(Tgrid,Data,kind='linear',axis=-1,copy=False,bounds_error=False,fill_value=np.mean(Data))
    Data=f(Tgrid-Tshift)
    return Data



def cubic(Data,Fs,Tshift):
    
    tdim=Data.ndim-1
    n=Data.shape[-1]
    Tgrid=np.array(range(n))/Fs
    
    f=interp1d(Tgrid,Data,kind='cubic',axis=-1,copy=False,bounds_error=False,fill_value=np.mean(Data))
    Data=f(Tgrid-Tshift)
    return Data


def spline(Data,Fs,Tshift):
    
    tdim=Data.ndim-1
    n=Data.shape[-1]
    Tgrid=np.array(range(n))/Fs
    
    f=interp1d(Tgrid,Data,kind='quadratic',axis=-1,copy=False,bounds_error=False,fill_value=np.mean(Data))
    Data=f(Tgrid-Tshift)
    return Data

def loadNIIimg(Fname):
    img=nb.load(Fname).get_data()
    return img


def RunAll(Fname,Fs,Int):
    Tr=1./Fs
    global Im
    Im=loadNIIimg(Fname)
    Lt = Im.shape[3]
    Nz = Im.shape[2]
    Ny = Im.shape[1]
    Nx = Im.shape[0]
    #print str(Nz)
    SliceOrder, SliceShift = MakeIntTime.MakeT(Int, 0, Nz, Tr)
    Slices = np.array(range(len(SliceShift)))
    FsIn=np.resize(Fs,Slices.shape)

    #print str(FsIn.shape)
    #print str(SliceShift.shape)
    
    inputs=np.column_stack((Slices,FsIn, SliceShift))
        
    
    p=Pool(Nz)
    SincSig = p.map(Sinc2,inputs)
    #print('Sinc Done')
    #print np.array(SincSig).shape
    LinSig = p.map(linear2,inputs)
    #print('Lin Done')
    #print np.array(LinSig).shape
    CubSig = p.map(cubic2,inputs)
    #print('Cubic Done')
    #print np.array(CubSig).shape
    SplSig = p.map(spline2,inputs)
    #print('SplineDone')
    #print np.array(SplSig).shape
    p.close()
    p.join()
    
    SincSig= np.swapaxes(np.swapaxes(np.array(SincSig),0,1),1,2)
    LinSig = np.swapaxes(np.swapaxes(np.array(LinSig),0,1),1,2)
    CubSig = np.swapaxes(np.swapaxes(np.array(CubSig),0,1),1,2)
    SplSig = np.swapaxes(np.swapaxes(np.array(SplSig),0,1),1,2)
    
    temp=nb.load(Fname)
    #print('making nifti')
    SincImg=nb.Nifti1Image(SincSig,temp.get_qform(),temp.get_header())
    LinImg=nb.Nifti1Image(LinSig,temp.get_qform(),temp.get_header())
    CubImg=nb.Nifti1Image(CubSig,temp.get_qform(),temp.get_header())
    SplImg=nb.Nifti1Image(SplSig,temp.get_qform(),temp.get_header())
    #print('done')

    
    
    return(SincImg,LinImg,CubImg,SplImg)
    
    
    

    
    
    
    

















    
