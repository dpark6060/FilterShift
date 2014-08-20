
import numpy as np
import nibabel as nb
import glob


def Cat(TruFiles,saveFile,noise,itlv):
    Bold20=np.empty([112,112,37,0])
    TruFiles.sort()
    for tru in TruFiles:
        image=nb.load(tru)
        Bold=image.get_data()
        del image
        Bold20=np.concatenate((Bold20,Bold),axis=-1)


    Bold_5=Bold20[:,:,:,::40]


	

    path,fil=os.path.split(tru)
    fil=path+'/SimulatedBoldSmooth_66_'+noise+'_Interleaved'+itlv+'.nii.gz'
    bad=nb.load(fil)
    BoldNii=nb.nifti1.Nifti1Image(Bold_5,bad.get_affine(),header=bad.get_header())
    nb.loadsave.save(BoldNii,saveFile)    


if __name__=="__main__":
    Cat(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
    
