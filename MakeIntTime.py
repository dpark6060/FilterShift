
import numpy as np
import sys

def MakeT(Int,Start,NumSlices,Tr):
	dt=float(Tr)/float(NumSlices)
	IntSeq=[]
	
	for i in range(Int):
		IntSeq=IntSeq+range(i,NumSlices,Int)
		
	IntSeq=np.array(IntSeq)
	IntSeq=np.roll(IntSeq,Start,0)
	IntTime=IntSeq*dt
	
	return(IntSeq,IntTime)
	
if __name__=="__main__":
	#Use: MakeIntTime <Interleave> <StartSlice> <NumSlices> <Tr>	
	IntS,IntT=MakeT(sys.argv[0],sys.argv[1],sys.argv[2],sys.argv[3])
	