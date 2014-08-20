#!/usr/bin/env python

import numpy as np
import sys
import matplotlib.pyplot as pl
import ZeroNorm

def SetPlot(ax,Data,color,line,marker,size,label):
    print Data.shape
    ax.plot(ZeroNorm.Normalize(ZeroNorm.Zero(Data)),linestyle=line,color=color,marker=marker,linewidth=size,label=label)
    return ax
def ShowPlot(ax,xmin,xmax,savedir):
    pl.xlim(xmin,xmax)
    pl.xlabel('Samples')
    pl.ylabel('magnitude')
    pl.legend()
    pl.savefig(savedir)
    #pl.show()
    pl.close()
    

def PlotFromImage(savedir,Nx,Ny,Nz,xmin,xmax,*args):
    f=pl.figure()
    ax=f.add_subplot(111)
    colors=[(100./255., 150./255., 255./255.),(168./255., 196./255., 255./255.),(200./255., 0., 0.),(0., 0., 1.),(20./255., 150./255., 70./255.),(222./255., 125./255., 0.)]
    
    
    #ColorBold=[100/255 150/255 255/255];
    #FadedBold=[168/255 196/255 255/255];
    #Color1=[200/255 0 0];
    #Color2=[0 0 1];
    #Color3=[20/255 150/255 70/255];
    #Color4=[222/255 125/255 0];
    
    for i in range(0,len(args),2):
        ax=SetPlot(ax,np.squeeze(args[i][Nx,Ny,Nz,:]),colors[i/2],'.',1,args[i+1])
        
    ShowPlot(ax,xmin,xmax,savedir)
    
    
    
    pass



if __name__=='__main__':
    f=pl.figure()
    ax=f.add_subplot(111)
    ax=SetPlot(ax,sys.argv[1].sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])
    pl.show()
    
