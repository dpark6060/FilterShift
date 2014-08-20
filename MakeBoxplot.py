#!/usr/bin/env python

import os
import glob
import nibabel as nb
import numpy as np
import matplotlib.pyplot as pl


def ReadStatFile(StatFile,savedir,plotName):
    Stats=open(StatFile)
    Stats=Stats.readlines()
    Stats=Stats[1:]
    
    meth=[]
    mserr=[]
    mxerr=[]
    
    for stat in Stats:
        meth=meth+[stat.split()[0].rstrip()]
        mserr=mserr+[float(stat.split()[1].rstrip())]
        mxerr=mxerr+[float(stat.split()[2].rstrip())]
        
    Box(np.array(meth),np.array(mserr),np.array(mxerr),savedir,plotName)
    
    
def Box(meth,mserr,mxerr,savedir,plotName):
    
    
    plots=np.unique(meth)
    mseStat=[]
    mxeStat=[]
    for plt in plots:
        if plt!='Spline':
            mseStat=mseStat+[mserr[np.where(meth==plt)[0]]]
            mxeStat=mxeStat+[mxerr[np.where(meth==plt)[0]]]
    
    plots=plots[np.where(plots!='Spline')]
    mseStat=np.array(mseStat).T
    mxeStat=np.array(mxeStat).T
    pl.figure(0)
    pl.boxplot(mseStat)
    pl.xlabel('Method')               
    pl.ylabel('Mean Squared Error')
    pl.title(plotName)
    pl.ylim(0,1.2)
    pl.xticks(np.array(range(len(plots)))+1,plots)     
    pl.savefig(savedir)
    pl.close()
     
