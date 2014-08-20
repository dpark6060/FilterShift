#!/usr/bin/env python

import os
import glob
import nibabel as nb
import CatBold
import MakeBoxplot as box
import numpy as np




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




#Subjects=open('/home/dparker/Documents/SubjectList.txt','r')
WrkDr='/home/dparker/Desktop/supercomputer_home/Desktop/FilterShiftAnalysis/'
Int=6
itlv=str(Int)
Target='Stats.txt'
Fs=0.5
savedir='Analysis/'


Header=['Method','MSE','Max Err']

Noises=['Noise0','Noise5','Noise10','Noise20','Noise40','Noise60']
meth=[]
mserr=[]
mxerr=[]



for noise in Noises:

    Header=['Method','MSE','Max Err']    
    meth=[]
    mserr=[]
    mxerr=[]
    Subjects=glob.glob(WrkDr+'DavidSub_*')

    for sub in Subjects:
    	tr,subid=os.path.split(sub)
	print subid
	StatFile=WrkDr+subid+'/'+subid+'_'+noise+'_Corrected/'+subid+Target
	print StatFile
	
	if os.path.exists(StatFile):
	    Stats=open(StatFile)
	    Stats=Stats.readlines()
	    Stats=Stats[1:]
	    
	    for stat in Stats:
		meth=meth+[stat.split()[0].rstrip()]
		mserr=mserr+[float(stat.split()[1].rstrip())]
		mxerr=mxerr+[float(stat.split()[2].rstrip())]
	 
    data=np.vstack((mserr,mxerr)).T 
    StatSave=WrkDr+savedir+'/'+noise+'_Stats.txt'
    PrintReport(StatSave,Header,meth,data)
    filename=WrkDr+savedir+'/'+noise+'_Stats.pdf'
    box.Box(np.array(meth),np.array(mserr),np.array(mxerr),filename,'MSE '+noise)
    
    
    
    
    


