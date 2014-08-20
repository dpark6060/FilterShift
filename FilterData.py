import sys

import DesignFilter
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as pl

# Use: FilterData <Sig>,<Filter>

def lowpass(CutOffFreq, SamplingRate, StopGain, TranWidth):

    NiquistRate = SamplingRate/2.0
   
    N, beta = sig.kaiserord(StopGain,TranWidth/NiquistRate)
    print 'the order of the FIR filter is:' + str(N) + ', If this is bigger than the size of the data please adjust the width and gain of the filter'
        
    taps = sig.firwin(N, CutOffFreq, window=('kaiser', beta), pass_zero=True, scale=True, nyq=NiquistRate)
    return (taps,NiquistRate)

def bandstop(CutOffFreq, SamplingRate, StopGain, TranWidth):
    NiquistRate = SamplingRate/2.0
    N, beta = sig.kaiserord(StopGain,TranWidth/NiquistRate)
    print 'the order of the FIR filter is:' + str(N) + ', If this is bigger than the size of the data please adjust the width and gain of the filter'
    taps = sig.firwin(N, CutOffFreq, window=('kaiser', beta), pass_zero=True, scale=True, nyq=NiquistRate)
    return (taps,NiquistRate)

        
def bandpass(CutOffFreq, SamplingRate, StopGain, TranWidth):
    NiquistRate = SamplingRate/2.0
    N, beta = sig.kaiserord(StopGain,TranWidth/NiquistRate)
    print 'the order of the FIR filter is:' + str(N) + ', If this is bigger than the size of the data please adjust the width and gain of the filter'
    taps = sig.firwin(N, CutOffFreq, window=('kaiser', beta), pass_zero=False, scale=True, nyq=NiquistRate)

    return (taps,NiquistRate)

def highpass(CutOffFreq, SamplingRate, StopGain, TranWidth):
    NiquistRate = SamplingRate/2.0
    N, beta = sig.kaiserord(StopGain,TranWidth/NiquistRate)
    print 'the order of the FIR filter is:' + str(N) + ', If this is bigger than the size of the data please adjust the width and gain of the filter'
        
    taps = sig.firwin(N, CutOffFreq, window=('kaiser', beta), pass_zero=False, scale=True, nyq=NiquistRate)   
    return (taps,NiquistRate)

def filter(Sig,Filter):
    LenOfTimeSeries=Sig.shape[-1]
    
    return sig.filtfilt(Filter, [1], Sig, axis=-1,padtype='even', padlen=0)


def PlotFilt(b,nyq_rate):
    a=[1];
    pi=np.pi
    pl.figure(1)
    w,h = sig.freqz(b,a)
    pl.subplot(311)
    pl.plot((w/pi)*nyq_rate, abs(h), linewidth=2)
    pl.xlabel('Frequency (Hz)')
    pl.ylabel('Gain')
    pl.title('Frequency Response')
    pl.ylim(-0.05, 1.05)
    pl.grid(True)
    
    pl.subplot(312)
    h_dB = 20 * np.log10 (abs(h))
    pl.plot((w/pi)*nyq_rate,h_dB)
    pl.xlabel('Frequency (Hz)')
    pl.ylabel('Gain')
    pl.title('Frequency Response')
    
    pl.subplot(313)
    h_Phase = np.unwrap(np.arctan2(np.imag(h),np.real(h)))
    pl.plot((w/pi)*nyq_rate,h_Phase)
    pl.grid()
    pl.ylabel('Phase (radians)')
    pl.xlabel( 'Frequency ')
    pl.title(r'Phase response')
    pl.show()
    pl.grid()


    a=1
    pl.figure(2)
    impulse = np.repeat(0.,50); impulse[0] =1.
    x = np.arange(0,50)
    response = sig.lfilter(b,a,impulse)
    pl.subplot(211)
    pl.stem(x, response)
    pl.ylabel('Amplitude')
    pl.xlabel(r'n (samples)')
    pl.title(r'Impulse response')
    pl.subplot(212)
    step = np.cumsum(response)
    pl.stem(x, step)
    pl.ylabel('Amplitude')
    pl.xlabel(r'n (samples)')
    pl.title(r'Step response')
    pl.subplots_adjust(hspace=0.5)
    pl.show()

if __name__=="__main__":
    main()
