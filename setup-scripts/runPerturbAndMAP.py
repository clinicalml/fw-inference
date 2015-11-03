#Run perturbMAP & save to MAT
import ipdb
import os,re,sys
from scipy.io import *
import numpy as np
sys.path.append('../source')
import perturbMAP as inferenceAlg
import util,timeit
#Run perturbMAP
#folder : Root folder
def runPerturbAndMAP(folder,numSamples):
	#raw_input("WARNING: PerturbAndMAP ONLY TESTED FOR BINARY RANDOM VARIABLES. PRESS A KEY TO CONTINUE")
	print "Running perturbMAP"
	inputfolder  = folder + '/uai_mat'
	matfolder = folder+'/mat_out'
	print inputfolder,matfolder
	if not os.path.exists(inputfolder):
		assert "Input folder not found. Investigate"
	if not os.path.exists(matfolder):
		os.mkdir(matfolder)
	for root, dirs, files in os.walk(inputfolder): # Walk directory tree
			for f in files:
				if f.endswith("mat"):
					print "Processing : ",f
					basename = re.split('.uai',f)[0]
					inputfilename = root+'/'+f
					N,Nodes,Edges,Cardinality,mode = util.loadUAIfromMAT(inputfilename)
					G = util.Graph(N=N,Edges=Edges,Nodes=Nodes,Cardinality=Cardinality,
							mode=mode,weighted=True,type='undir')
					start = timeit.default_timer()
					logZ,mus,time_marg,time_logz = inferenceAlg.doInferencePerturbMAP(G,numSamples)
					stop = timeit.default_timer()
					print stop-start," seconds taken"
					print "Saving marginals: ",mus
					assert mus.shape[0]==np.sum(G.Cardinality),"Marginal vector not as expected"
					matfile= matfolder+'/'+basename
					if os.path.exists(matfile):
						mat = loadmat(matfile)
						print "Appending ",matfile
					else:
						mat = {}
						print "Creating new mat file, ",matfile
					mat['perturbMAP_node_marginals_ub'] = np.reshape(mus,(len(mus),1))
					mat['perturbMAP_log_partition_ub'] = logZ
					mat['perturbMAP_runtime_ub_logz'] = time_logz
					mat['perturbMAP_runtime_ub_marg'] = time_marg
					mat['perturbMAP_runtime_ub_total'] = stop-start
					savemat(matfile, mat)
