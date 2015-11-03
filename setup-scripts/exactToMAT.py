import numpy as np
from scipy.io import *
import os

#Input file with outpout of JT converted to MAT format
#folder: Root folder 
def exactToMAT(folder):
	inputfolder = folder+'/exact_out'
	outputfolder = folder+'/mat_out'

	if os.path.exists(outputfolder)==False: 
		os.mkdir(outputfolder)
	if os.path.exists(inputfolder)==False:
		print "Error. Input directory not found"
	for root, dirs, files in os.walk(inputfolder): # Walk directory tree
			for f in files:
				if f.endswith("out"):
					inputfile = root+'/'+f
					f = f.replace(".0","")
					outputfile= outputfolder+'/'+f.strip().split('.out')[0]+'.mat'
					print 'Reading from: ',inputfile
					mode = 0
					log_part = -3.1415
					mus = []
					for line in open(inputfile):
						if "Exact variable marginals:" in line:
							#start collecting data
							mode = 1
							continue
						if "Exact factor marginals:" in line:
							#done collecting 
							mode = 2
						if "Exact log partition sum:" in line:
							log_part = float(line.strip().split(":")[1])
							break
						#collecting unary marginals
						if mode == 1:
							elems = line.strip().split(',')
							clean_elems = [e.replace("(","").replace(")","").replace("{","").replace("}","") for e in elems]
							for i in xrange(1,len(clean_elems)):
								mus.append(float(clean_elems[i]))
						#collecting factor marginals
						if mode ==2:
							continue
					print 'Writing to:',outputfile
					mus = np.array(mus)
					print "Unary Marginals Shape: ",mus.shape, " Log Partition :",log_part
					if os.path.exists(outputfile):
						mat = loadmat(outputfile)
						print "Appending"
					else:
						mat = {}
						print "Creating new mat file"
					mat['exact_node_marginals'] = np.reshape(mus,(max(mus.shape),1))
					mat['log_partition'] = log_part
					savemat(outputfile, mat,appendmat=True)
	
if __name__ == '__main__':
	exactToMAT('../Trees')