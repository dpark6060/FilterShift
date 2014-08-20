
import sys
import nibabel as nb
import numpy as np
import subprocess as sp
import scipy.signal as sig
import pylab as pl



def lowpass(CutOffFreq, SamplingRate, StopGain, TranWidth):

    NiquistRate = SamplingRate/2.0
   
    N, beta = sig.kaiserord(StopGain,TranWidth/NiquistRate)
    print 'the order of the FIR filter is:' + str(N) + ', If this is bigger than the size of the data please adjust the width and gain of the filter'
        
    taps = sig.firwin(N, CutOffFreq, window=('kaiser', beta), pass_zero=True, scale=True, nyq=NiquistRate)
    return taps

def bandstop(CutOffFreq, SamplingRate, StopGain, TranWidth):
    NiquistRate = SamplingRate/2.0
    N, beta = sig.kaiserord(StopGain,TranWidth/NiquistRate)
    print 'the order of the FIR filter is:' + str(N) + ', If this is bigger than the size of the data please adjust the width and gain of the filter'
    taps = sig.firwin(N, CutOffFreq, window=('kaiser', beta), pass_zero=True, scale=True, nyq=NiquistRate)
    return taps

        
def bandpass(CutOffFreq, SamplingRate, StopGain, TranWidth):
    NiquistRate = SamplingRate/2.0
    N, beta = sig.kaiserord(StopGain,TranWidth/NiquistRate)
    print 'the order of the FIR filter is:' + str(N) + ', If this is bigger than the size of the data please adjust the width and gain of the filter'
    taps = sig.firwin(N, CutOffFreq, window=('kaiser', beta), pass_zero=False, scale=True, nyq=NiquistRate)

    return taps

def highpass(CutOffFreq, SamplingRate, StopGain, TranWidth):
    NiquistRate = SamplingRate/2.0
    N, beta = sig.kaiserord(StopGain,TranWidth/NiquistRate)
    print 'the order of the FIR filter is:' + str(N) + ', If this is bigger than the size of the data please adjust the width and gain of the filter'
        
    taps = sig.firwin(N, CutOffFreq, window=('kaiser', beta), pass_zero=False, scale=True, nyq=NiquistRate)   
    return taps


