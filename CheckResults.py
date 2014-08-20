#!/usr/bin/env python

import numpy as np
import ZeroNorm

def Error(Corrected,Tru):
    
    Corrected=ZeroNorm.Normalize(ZeroNorm.Zero(Corrected))
    Tru=ZeroNorm.Normalize(ZeroNorm.Zero(Tru))
    
    MSE=np.sum(np.square(np.subtract(Corrected,Tru)))/(np.nonzero(np.sum(Tru,axis=3))[1].shape[-1]*Tru.shape[-1])
    MAXERR=np.amax(np.square(np.subtract(Corrected,Tru)))
    
    return (MSE,MAXERR)

    
