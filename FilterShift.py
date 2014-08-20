#!/usr/bin/env python

import scipy as sp
import numpy as np,sys
import nibabel as nb
import scipy.signal as sig
import matplotlib.pyplot as pl
from multiprocessing import Pool
import multiprocessing
import time
import sys
import FilterData


def MakeT(Int,Start,NumSlices,Tr):
# Creates timing sequence for the slices in a given image.
# Int - the interpolation value (1 - sequential, 2 - Odd/Even, 3 - 1,4,7,... 2,5,8,... 3,6,9..., etc)
# Start - the slice number to align to. 0 indicates the first slice.
# NumSlices - the total number of slices
# Tr - the TR of the data, or the sampling rate.

# Calculate dt: the time interval between any two slices acquired sequentially
    dt=float(Tr)/float(NumSlices)
    IntSeq=[]
    
# Create slice acquisition order sequence
    for i in range(Int):
        IntSeq=IntSeq+range(i,NumSlices,Int)    
    IntSeq=np.array(IntSeq)

# Initialize slice timing array
    IntTime=np.zeros(IntSeq.size)
    IntTime[0]=0
    
# Look search for the location of a slice in the acquisition order sequence, and multiply by dt
# For example, if slice 2 was the 14th slice to be acquired, its index in IntSeq would be 13, so
# the time of that slice would be 13*dt.  13 is used instead of 14 because the first slice is
# acquired at time T=0, and slice 2 was acquired 13 slices AFTER slice 1.
    for i in range(1,NumSlices):
    	IntTime[i]=np.nonzero(IntSeq==i)[0][0]*dt

# Zero the acquisition time around the slice of interest    
    IntTime=IntTime-IntTime[Start]
    return(IntSeq,IntTime)


##################################################################################################
##################################################################################################
## Filter Construction functions:
##
## BANDPASS: Allows for the creation of a kaiser window bandpass filter
## CutOffFreq: [fstop1, fstop2]
## SamplingRate: sampling frequency of data
## StopGain: the gain of the stopband, where the stopband gain = -1*StopGain dB
## TranWidth: the transition width of the filter centered around the cutoff frequency.
## For example, if Fcutoff = 2hz, and TranWidth = 0.8, the transition band will be 1.6hz - 2.4 hz 
##
##
## LOWPASS: Allows for the creation of a kaiser window lowpass filter
## CutOffFreq: [fstop1]
## SamplingRate: sampling frequency of data
## StopGain: the gain of the stopband, where the stopband gain = -1*StopGain dB
## TranWidth: the transition width of the filter centered around the cutoff frequency.
## For example, if Fcutoff = 2hz, and TranWidth = 0.8, the transition band will be 1.6hz - 2.4 hz 
##
##################################################################################################
##################################################################################################

def bandpass(CutOffFreq, SamplingRate, StopGain, TranWidth):
    NiquistRate = SamplingRate/2.0
    N, beta = sig.kaiserord(StopGain,TranWidth/NiquistRate)
    taps = sig.firwin(N, CutOffFreq, window=('kaiser', beta), pass_zero=False, scale=True, nyq=NiquistRate)
    return (taps,NiquistRate)

def lowpass(CutOffFreq, SamplingRate, StopGain, TranWidth):
    NiquistRate = SamplingRate/2.0   
    N, beta = sig.kaiserord(StopGain,TranWidth/NiquistRate)        
    taps = sig.firwin(N, CutOffFreq, window=('kaiser', beta), pass_zero=True, scale=True, nyq=NiquistRate)
    return (taps,NiquistRate)


def loadNIIimg(Fname):
    img=nb.load(Fname).get_data()
    return img



def FiltShift(ZTshift):
# The main filtershift function, takes a single vector ZTshift
# ZTshift: [Slice Number, Time Shift]
# where Slice Number is the index of the slice to be shifted, and Time Shift is the amount to shift.  This can be positive or negative.

    Z=ZTshift[0]
    tShift=ZTshift[1]

    Ny=45

# The slice is extracted from the 4D global image and reshaped into a 3D volume, representing the 2D slice and its time series.
    Sig=np.squeeze(Im[:,:,Z,:])
    
# The time axis must be the last dimension.
    tdim=len(list(np.shape(Sig)))-1    
    
# Length Time Sig (LTS) is the length of the time series associated with the slice signal.
    LTS=Sig.shape[-1]

# Padding is added to the front end end of the slice, where half the signal is mirrored on each end
# FR - Front Range: the range of indicies to be padded on the front (beginning) of the signal
# FR starts at 1 and ends at LTS/2. (0 is the first index)
# BR - Back Range: the range of indicies to be padded on the back (end) of the signal
# BR starts at LTS-1 and goes to LST/2
    FR=np.array(range(int(round(LTS/2.)),0,-1))
    BR=np.array(range(LTS-1,int(FR.shape[-1])-1,-1))


# Pad the signal, with conditions for each dimension so that all data shapes can be accomodated. 
    if tdim==0:
    # One dimensional case (Single Vector)
    
# The signal to be padded on the front is the values of Sig at indecies FR
# The signal to be padded to the     
        FrontPad=(Sig[FR])	
        BackPad=Sig[BR]

# The length of the padding is stored as LFP
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

# The padding is added to the signal along the time axis
    Sig=np.concatenate((FrontPad,Sig,BackPad),-1)

# Upsampling/interpolation paramaters are calculated
# S: Upsampling Factor, or the number of samples to be added between existing samples
# Dm: the dimensions that the upsampled signal will be
# SS: the dimensions of the padded signal
    S=int(round(Fnew/Foriginal))
    Dm=list(np.shape(Sig))
    SS=Dm
    tdim=len(Dm)-1
    Dm[tdim]=Dm[tdim]*S

# Initialize upsampled signal as zeros
    ZeroSig=np.zeros(Dm)

    k=0
    
# Create Zero Padded Signal    
    if tdim==0:
        
# Assign every S samples in ZeroSig to values in Sig    
        ZeroSig[::S]=Sig
    elif tdim==1:       
        ZeroSig[:,::S]=Sig
    elif tdim==2:        
        ZeroSig[:,:,::S]=Sig                 
    elif tdim==3:       
        ZeroSig[:,:,:,::S]=Sig
    else:
        print("Bad Array Dimensions")  

# Cleanup Sig as it's no longer needed   
    del Sig    

# Filter the Zero padded signal with the designed filter
    ZeroSig = sig.filtfilt(BPF, [1], ZeroSig*float(S), axis=-1,padtype='even', padlen=0) 

# Calculate new frequency and time parameters for the upsampled signal
    tdim=len(SS)-1
    Fs=Fnew
    Ts=1/Fs

# Initialize a variable the length of the padded signal at the original frequency
    Sig=np.zeros(SS)
    
# Calculate the number of indicies to shift when resampling    
    shift=round(tShift*Fs)

# Shift the Signal
    if tdim==0:
    # One Dimensional Case (1D vector)
    
        if shift>0:
        # If the shift is larger than zero
        # Extend the Upsampled signal by repeating the values at the beginning of the signal by the shift amount
        # then resample the signal, starting at index 0, every S indecies, to one S of the end
            Rep=np.tile(ZeroSig[0],shift)
            ZeroSig=np.append(Rep,ZeroSig,-1)
            Sig=ZeroSig[range(0,ZeroSig.shape[-1]-S,S)]       
        
        else:
        # If the Shift is less than zero
        # Extend the Upsampled signal by repeating the values at the end of the signal by the shift amount
        # Then resample the signal, starting at index shift, every s indicies, to the end
            Rep=np.tile(ZeroSig[-1],abs(shift))
            ZeroSig=np.append(ZeroSig,Rep,-1)
            Sig=ZeroSig[range(int(abs(shift)),ZeroSig.shape[-1]-1,S)]
        
        # Crop the signal to remove the padding preformed earlier    
        Sig=Sig[LFP:LFP+LTS]
        
    elif tdim==1:  
     
        if shift>0:
            Rep=np.tile(np.expand_dims(ZeroSig[:,0],axis=-1),[1,shift])
            ZeroSig=np.append(Rep,ZeroSig,-1)
            Sig=ZeroSig[:,range(0,ZeroSig.shape[-1]-S,S)]       
    
        else:
            Rep=np.tile(np.expand_dims(ZeroSig[:,-1],axis=-1),[1,abs(shift)])
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


    return Sig
   
       
def Run(Fname,Fs,Int):
# Run the slice timing correction on a particular file
# Fname: the location of the file to be used
# Fs: the sampling frequency of the file (Tr)
# Int - the interpolation value (1 - sequential, 2 - Odd/Even, 3 - 1,4,7,... 2,5,8,... 3,6,9..., etc)


#Fname = "/Data/DavidSub_4/SimulatedBoldSmooth_66_Noise0_Interleaved6.nii.gz"

# Declare Global Variables to be used in other functions
# Im: the full 4D fMRI image 
# Fnew: The frequency to upsample to
# Foriginal: the original sampling frequency, in this case Fs
# BPF: a filter to be designed in program       
    global Im
    global Fnew
    global Foriginal
    global BPF

# Load the file
    Im = loadNIIimg(Fname)

# Gather information on the image file
    Lt = Im.shape[3]
    Nz = Im.shape[2]
    Ny = Im.shape[1]
    Nx = Im.shape[0]
    
# Set sampling information  
    Tr= 1./Fs
    Foriginal = Fs # Hz
    Fnew = 20.0 #Hz
    Stopgain = 60 # -1*dB
    Tranwidth = 0.08 # Hz

# Create lowpass Filter
    BPF, Nyq = lowpass(0.2, Fnew, Stopgain, Tranwidth)

# Create vectors for the order of slice acquisition (SliceOrder) and the shift required to align each slice to RefSlice (SliceShift)
    RefSlice=39
    SliceOrder, SliceShift = MakeT(Int, RefSlice, Nz, Tr)
    Slices = np.array(range(len(SliceShift)))
    
# Create inputs for multipool process    
    inputs = np.column_stack((Slices, SliceShift))   

# Initialize Multipool
    p=Pool(1)

# Run the filtering process
    FiltSigs = p.map(FiltShift, inputs)
    
    FiltSigs=np.array(FiltSigs)
    FiltSigs=np.swapaxes(np.swapaxes(FiltSigs,0,1),1,2)
    temp=nb.load(Fname)    
    img=nb.Nifti1Image(FiltSigs,temp.get_affine(),temp.get_header())
    return img

if __name__=="__main__":
	#print(sys.argv[1])
	Run(sys.argv[1],sys.argv[2],sys.argv[3])
# Syntax: python FilterShift.py Fname Fs Int
# Fname: the location of the file to be used
# Fs: the sampling frequency of the file (Tr)
# Int - the interpolation value (1 - sequential, 2 - Odd/Even, 3 - 1,4,7,... 2,5,8,... 3,6,9..., etc)

