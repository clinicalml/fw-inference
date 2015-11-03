#Source code for MAP solver using toulbar2
import numpy as np
import os,subprocess,util,ipdb

#Run MAP inference given a marginal vector
#IMPORTANT: Assumes potentials in marg_vec are in *LOG* format
def runMAP(marg_vec,N,List,Cardinality,graphType,uai_fname = './temp.uai',timer=-1):
	#TOULBAR_BIN = '/home/rahul/Research/opt/fw-python/toulbar2/bin/toulbar2-mod'
	TOULBAR_BIN = '../toulbar2.0.9.7.0-Release-sources/build/bin/Linux/toulbar2'
	if not os.path.exists(TOULBAR_BIN):
		assert False,"toulbar2 binary not found at :"+TOULBAR_BIN

	#write evidence file
	f = open(uai_fname+'.evid','w')
	f.write('0')
	f.close()

	#write file
	#Convert marg_vec to potentials

	nEdges = List.shape[0]
	EdgeList = List[:nEdges,:2].astype(int)
	util.writeToUAI(uai_fname,np.exp(marg_vec),N,Cardinality,EdgeList)

	soln_fname = uai_fname.split('.uai')[0]+'.sol'

	#Run MAP inference
	cmd = TOULBAR_BIN + ' '+uai_fname+' -w='+soln_fname
	if timer>0:
		cmd += ' -timer='+str(timer)

	#Must run from run directory
	run_dir = uai_fname.rsplit('/',1)[0]
	run_dir += '/'
	result = subprocess.check_output(cmd,stderr=subprocess.STDOUT, shell=True)
	#Write stdout to logfile
	resultfile = uai_fname.split('.uai')[0]+'.log'
	f_log = open(resultfile,'w')
	f_log.write(result+'\n')
	f_log.close()

	#Get solution
	MAPsoln = np.loadtxt(soln_fname)

	os.unlink(soln_fname)
	if os.path.exists(soln_fname):
		assert False,"sol file not deleted"
	os.unlink(uai_fname+'.evid')
	os.unlink(uai_fname)
	assert max(MAPsoln.shape)==max(Cardinality.shape),"All vertices not present in MAP solution"+str(MAPsoln.shape)+" vs "+str(Cardinality.shape)
	vertex_marg_polytope= []

	for i in xrange(N):
		v_card = Cardinality[i]
		for j in xrange(v_card):
			if j==MAPsoln[i]:
				vertex_marg_polytope.append(1)
			else:
				vertex_marg_polytope.append(0)

	for ei in xrange(EdgeList.shape[0]):
		v1 = EdgeList[ei,0]
		v2 = EdgeList[ei,1]
		card_edge = Cardinality[v1]*Cardinality[v2]
		ctr = 0
		for i in xrange(Cardinality[v1]):
			for j in xrange(Cardinality[v2]):
				if MAPsoln[v1]==i and MAPsoln[v2]==j:
					vertex_marg_polytope.append(1)
				else:
					vertex_marg_polytope.append(0)
				ctr +=1
		assert ctr == card_edge,"Edge cardinality does not match up"+str(ctr)+str(card_edge)
	return np.array(vertex_marg_polytope),result,MAPsoln
