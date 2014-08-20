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
import SigInterp
import CheckResults
import PlotData



def SaveNii(fName,nii):
    nb.loadsave.save(nii,fName)


def PrintReport(Fname,header,col,data):
    f=open(Fname,'w')
    hdr=''
    for i in range(len(header)-1):
        hdr=hdr+header[i]+'\t'
        
    hdr=hdr+header[-1]+'\n'
    f.write(hdr)
    f.close()
    
    f=open(Fname,'a')
    for i in range(data.shape[0]):
        MyStr=col[i]+'\t'
        for k in range(data.shape[-1]-1):
            
            MyStr=MyStr+'%06E\t' % (data[i,k])
            
        MyStr=MyStr+'%06E\n' % (data[i,-1])
        f.write(MyStr)
    
    f.close()


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

def lowpass(CutOffFreq, SamplingRate, StopGain, TranWidth):

    NiquistRate = SamplingRate/2.0
   
    N, beta = sig.kaiserord(StopGain,TranWidth/NiquistRate)
    print 'the order of the FIR filter is:' + str(N) + ', If this is bigger than the size of the data please adjust the width and gain of the filter'
        
    taps = sig.firwin(N, CutOffFreq, window=('kaiser', beta), pass_zero=True, scale=True, nyq=NiquistRate)
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
    
    #print('Padding Finished')

    del Sig    
    print ZeroSig.shape[-1]
    #print('Filtering')
    ZeroSig = sig.filtfilt(BPF, [1], ZeroSig*float(S), axis=-1,padtype='even', padlen=0) 
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
   
       
#Fname = "/Data/DavidSub_4/SimulatedBoldSmooth_66_Noise0_Interleaved6.nii.gz"
    
TruFile='/home/dparker/Documents/Sub5_CleanBold.nii'
LoadFile='/Data/DavidSub_5/SimulatedBoldSmooth_66_Noise0_Interleaved6.nii.gz'
Fname=LoadFile
global Im
global Fnew
global Foriginal
global BPF
print(Fname)
Im = loadNIIimg(Fname)
print("Image Loaded")   
Lt = Im.shape[3]
Nz = Im.shape[2]
Ny = Im.shape[1]
Nx = Im.shape[0]
Tr = 2.0
Fs=0.5
Int=6    
Foriginal = Fs # Hz
Fnew = 20.0 #Hz
Stopgain = 60
Tranwidth = 0.08
print("Designing Filter")    
BPF, Nyq = lowpass(0.2, Fnew, Stopgain, Tranwidth)
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
FshiftImg=nb.Nifti1Image(FiltSigs,temp.get_qform(),temp.get_header())

(SincShift,LinShift,CubShift,SplShift)=SigInterp.RunAll(LoadFile,Fs,Int)
SincSave=saveprefix+'Sinc_'+saveName
LinSave=saveprefix+'Lin_'+saveName
CubSave=saveprefix+'Cubic_'+saveName
SplSave=saveprefix+'Spline_'+saveName
FshiftSave=saveprefix+'FilterShift_'+saveName



#print('Saving')
#nb.loadsave.save(SincShift,SincSave)
#nb.loadsave.save(FshiftImg,FshiftSave)
#nb.loadsave.save(LinShift,LinSave)
#nb.loadsave.save(CubShift,CubSave)
#nb.loadsave.save(SplShift,SplSave)
#print('Done')

Tru=nb.load(TruFile).get_data()
Orig=nb.load(LoadFile).get_data()

#PlotData.PlotFromImage('/home/dparker/Desktop/TestImg.png',54,63,13,5,200,Tru,'Underlying Bold',Orig,'Original',FshiftImg.get_data(),'FilterShifr',LinShift.get_data(),'Linear')
 
f=pl.figure()
ax=f.add_subplot(111)
#colors=[(100./255., 150./255., 255./255.),(168./255., 196./255., 255./255.),(200./255., 0., 0.),(0., 0., 1.),(20./255., 150./255., 70./255.),(222./255., 125./255., 0.)]
    
ColorBold=[100./255, 150./255, 255./255]
FadedBold=[168./255, 196./255, 255./255]
Color1=[200./255, 0, 0]
Color2=[0, 0, 1]
Color3=[20./255, 150./255, 70./255]
Color4=[222./255, 125./255, 0]
ColorBlack=[0, 0, 0]

Nx=54
Ny=60
Nz=15
xmin=50
xmax=125
savedir='/home/dparker/Desktop/TestImg.png'
   
ax=PlotData.SetPlot(ax,np.squeeze(Tru[Nx,Ny,Nz,:]),FadedBold,'-','',4,'Underlying Bold')
ax=PlotData.SetPlot(ax,np.squeeze(Orig[Nx,Ny,Nz,:]),ColorBlack,':','',1,'Original fMRI signal')
ax=PlotData.SetPlot(ax,np.squeeze(FshiftImg.get_data()[Nx,Ny,Nz,:]),Color1,'-','',1,'Filter Shift')
ax=PlotData.SetPlot(ax,np.squeeze(SincShift.get_data()[Nx,Ny,Nz,:]),Color2,'-','',1,'Sinc Interpolation')
ax=PlotData.SetPlot(ax,np.squeeze(LinShift.get_data()[Nx,Ny,Nz,:]),Color3,'-','',1,'Linear Interpolation')
ax=PlotData.SetPlot(ax,np.squeeze(CubShift.get_data()[Nx,Ny,Nz,:]),Color4,'-','',1,'Cubic Interpolation')

  
PlotData.ShowPlot(ax,xmin,xmax,savedir)
 
Header=['Method','MSE','Max Err']
Col=['FiltShift','Sinc','Linear','Cubic','Spline']
data=np.empty([5,2])
print('Checking Results')

mse,mxe=CheckResults.Error(FshiftImg.get_data(),Tru)    
data[0,0]=mse
data[0,1]=mxe
mse,mxe=CheckResults.Error(SincShift.get_data(),Tru)
data[1,0]=mse
data[1,1]=mxe
mse,mxe=CheckResults.Error(LinShift.get_data(),Tru)
data[2,0]=mse
data[2,1]=mxe
mse,mxe=CheckResults.Error(CubShift.get_data(),Tru)
data[3,0]=mse
data[3,1]=mxe

mse,mxe=CheckResults.Error(SplShift.get_data(),Tru)
data[4,0]=mse
data[4,1]=mxe
print('Done')

StatSave=saveprefix+"Stats.txt"
PrintReport(StatSave,Header,Col,data)






