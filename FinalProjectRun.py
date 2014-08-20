#!/usr/bin/env python

import os
import glob
import FilterShift
import nibabel as nb


Subjects=glob.glob('/home/dparker/Scans/FinalProject/FilterShift/P*/HRF*IntNorm*')
#print Subjects

for line in Subjects:
    print line
    path,filename=os.path.split(line)
    #print 
    #test=glob.glob(path+'/a*');
    #if not test:
    print('FilterShift.Run('+line+' 0.5, 6)')
    NiiImg=FilterShift.Run(line,0.5,6)
    print('nb.loadsave.save(NiiImg,'+path+'/a'+filename+')')
    nb.loadsave.save(NiiImg,path+'/a'+filename)
