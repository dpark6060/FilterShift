#!/usr/bin/env python

import scipy as sp
import numpy as np,sys
import nibabel as nb
import scipy.signal as sig
from scipy.interpolate import interp1d
import matplotlib.pyplot as pl
import FilterShift2 as FilterShift
import SigInterp
import sys
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

def RunOnFile(TruFile,LoadFile,saveprefix,saveName,Fs,Int):
    
    FshiftImg=FilterShift.Run(LoadFile,Fs,Int)
    #FshiftImg=nb.load('/home/dparker/Desktop/Sub4Processed.nii')
    #LinShift=nb.load('/home/dparker/Desktop/Analysis/Lin_Processed.nii')
    (SincShift,LinShift,CubShift,SplShift)=SigInterp.RunAll(LoadFile,Fs,Int)
    SincSave=saveprefix+'_Sinc_'+saveName
    LinSave=saveprefix+'_Lin_'+saveName
    CubSave=saveprefix+'_Cubic_'+saveName
    SplSave=saveprefix+'_Spline_'+saveName
    FshiftSave=saveprefix+'_FilterShift_'+saveName
    
   
    
    print('Saving')
    nb.loadsave.save(SincShift,SincSave)
    nb.loadsave.save(FshiftImg,FshiftSave)
    nb.loadsave.save(LinShift,LinSave)
    nb.loadsave.save(CubShift,CubSave)
    nb.loadsave.save(SplShift,SplSave)
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
    xmin=60
    xmax=100
    savedir=saveprefix+'_TestImg.png'
       
    ax=PlotData.SetPlot(ax,np.squeeze(Tru[Nx,Ny,Nz,:]),FadedBold,'-','',4,'Underlying Bold')
    ax=PlotData.SetPlot(ax,np.squeeze(Orig[Nx,Ny,Nz,:]),ColorBlack,':','',1,'Original fMRI signal')
    ax=PlotData.SetPlot(ax,np.squeeze(FshiftImg.get_data()[Nx,Ny,Nz,:]),Color1,'-','',1,'Filter Shift')
    ax=PlotData.SetPlot(ax,np.squeeze(SincShift.get_data()[Nx,Ny,Nz,:]),Color2,'-','',1,'Sinc Interpolation')
    ax=PlotData.SetPlot(ax,np.squeeze(LinShift.get_data()[Nx,Ny,Nz,:]),Color3,'-','',1,'Linear Interpolation')
    #ax=PlotData.SetPlot(ax,np.squeeze(CubShift.get_data()[Nx,Ny,Nz,:]),Color4,'-','',1,'Cubic Interpolation')    
      
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
    #print('Done')
    
    StatSave=saveprefix+"Stats.txt"
    PrintReport(StatSave,Header,Col,data)
   
if __name__=="__main__":
    #print (sys.argv[4])
    RunOnFile(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],float(sys.argv[5]),int(sys.argv[6]))
    
