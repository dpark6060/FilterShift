#!/usr/bin/env python

import numpy as np

def Zero(Data):
    
    tdim=Data.ndim-1    
    if tdim==0:
        Corrected=Data-Data.mean()
    elif tdim==1:
        Corrected=Data-np.tile(np.expand_dims(np.mean(Data,1),-1),[1,np.shape(Data)[-1]])
    elif tdim==2:
        Corrected=Data-np.tile(np.expand_dims(np.mean(Data,2),-1),[1,1,np.shape(Data)[-1]])
    elif tdim==3:
        Corrected=Data-np.tile(np.expand_dims(np.mean(Data,3),-1),[1,1,1,np.shape(Data)[-1]])
    else:
        print('Bad Array Dimension Size in Zero')
        
    return Corrected

def Normalize(Data):
    
    tdim=Data.ndim-1    
    if tdim==0:
        
        Scor=np.std(Data)
        if Scor==0:
            Scor=1
        Corrected=Data/Scor
        
    elif tdim==1:
        Scor=np.std(Data,1)
        Scor[np.nonzero(Scor==0)]=1
        Corrected=Data/np.tile(np.expand_dims(Scor,-1),[1,np.shape(Data)[-1]])
    elif tdim==2:
        Scor=np.std(Data,2)
        Scor[np.nonzero(Scor==0)]=1
        Corrected=Data/np.tile(np.expand_dims(Scor,-1),[1,1,np.shape(Data)[-1]])
    elif tdim==3:
        Scor=np.std(Data,3)
        Scor[np.nonzero(Scor==0)]=1
        Corrected=Data/np.tile(np.expand_dims(Scor,-1),[1,1,1,np.shape(Data)[-1]])
    else:
        print('Bad Array Dimension Size in Normalize')
        
    return Corrected