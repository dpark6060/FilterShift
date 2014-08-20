#!/usr/bin/env python

import os
import glob
import SliceTimingAnalysis
import nibabel as nb
import CatBold


#Subjects=open('/home/dparker/Documents/SubjectList.txt','r')
WrkDr='/home/dparker/Desktop/FilterShiftAnalysis/'
Int=6
itlv=str(Int)
truTarget='/SimulatedBoldSmooth_66_p*'
Suffix='Corrected'
Fs=0.5


Subjects=glob.glob('/data/DavidSub_*')
print Subjects

Noises=['Noise0','Noise5','Noise10','Noise20']

for noise in Noises:

    target='/SimulatedBoldSmooth_66_'+noise+'_Interleaved'+itlv+'.nii.gz'

    for sub in Subjects:
    
        TruFiles=glob.glob(sub+truTarget)
        TargetFiles=glob.glob(sub+target)

        pth,filename=os.path.split(sub)

        if (len(TruFiles)==4)&(len(TargetFiles)==1):
            SvDr=WrkDr+filename

	    if not os.path.exists(SvDr):
	        os.mkdir(SvDr)

	    SvDr=SvDr+'/'+filename+'_'+noise+'_Corrected'
            
	    if not os.path.exists(SvDr):
                os.mkdir(SvDr)
    
            TruBoldF=WrkDr+filename+'/'+filename+'_TruBold.nii'
            print(TruBoldF)
            if not(os.path.exists(TruBoldF)):
                CatBold.Cat(TruFiles,TruBoldF,noise,itlv)
    
            CorFilePrefix=SvDr+'/'+filename
    
            if not(os.path.exists(CorFilePrefix+'_FilterShift_Corrected.nii')):
            
                Cmd='python SliceTimingAnalysis.py '+TruBoldF+' '+TargetFiles[0]+' '+CorFilePrefix+' '+Suffix+' '+str(Fs)+' '+str(Int)
                #print(Cmd)
                os.system('echo '+Cmd)
		print ('Started at: ')
		os.system('date')
                os.system(Cmd)
		print('Finished at: ')
		os.system('date')
                #SliceTimingAnalysis.RunOnFile(TruBoldF,sub.rstrip()+target,CorFilePrefix,Suffix,Fs,Int)
            else:
                print(filename+' '+noise+' already processed')
        else:
            print('Skipping '+filename+', missing files')



