#Run MAP inference using opengm's tools
import numpy as np
import sys,os,util,subprocess,ipdb,time
from multiprocessing import Pool

#Parallel child function
def runPAR(args):
	name,cmd = args
	pid = os.getpid()
	print name," is running on pid=",pid
	start = time.time()
	os.system(cmd)
	end = time.time()
	print name," is finished on pid=",pid
	return name,end-start

#Run MAP inference
def runMAP(pot,N,List,
		Cardinality,graphType,
		uai_fname = './temp.uai',mapsolver = 'lazyflipper',initLabelling = None,logf = None):
	#ICM/Lazyflipper require an initial labelling
	if initLabelling is None and (mapsolver=='lazyflipper' or mapsolver == 'icm'):
		assert False,'icm/lazyflipper require initial labelling'
	
	CONVERTER_BIN = '../opengm-tools/uai2opengm'
	MAP_BIN = '../opengm-tools/map_solver'
	assert os.path.exists(CONVERTER_BIN),'Converter not found'
	assert os.path.exists(MAP_BIN),'map solver not found'
	assert mapsolver=='lazyflipper' or mapsolver=='qpbo' or mapsolver=='trws' or mapsolver=='all'\
			or mapsolver=='lsatr' or mapsolver=='dualdecomposition' or mapsolver=='icm'\
			,'Invalid map solver specified'

	fxn_start_marker = time.time()
	#Bookkeeping
	dir = os.path.dirname(uai_fname)
	base= os.path.basename(uai_fname)
	evidfile  = uai_fname+'.evid'
	logfile   = dir+'/'+base+'.log'
	solfile   = uai_fname+'.MPE'
	gm_fname = uai_fname.replace('uai','gm')
	initfname = uai_fname.replace('uai','init')
	outputfile= dir+'/'+base+'.out'

	#Write UAI file
	nEdges = List.shape[0]
	EdgeList = List[:nEdges,:2].astype('int')
	util.writeToUAI(uai_fname,np.exp(pot),N,Cardinality,EdgeList)
	f = open(evidfile,'w')
	f.write('0')
	f.close()

	#Convert to GM format
	cmd = CONVERTER_BIN + ' ' +uai_fname+ ' ' + gm_fname
	start =time.time()
	os.system(cmd)
	converter_time = time.time()-start

	#Run TRWS,QPBO and DualDecomposition in parallel and take
	#best result
	if mapsolver=='all':
		cmd_qpbo = MAP_BIN + ' ' +gm_fname + ' ' + solfile+'.qpbo' + ' ' + 'qpbo >'+logfile
		cmd_trws = MAP_BIN + ' ' +gm_fname + ' ' + solfile+'.trws' + ' ' + 'trws >'+logfile
		cmd_dd = MAP_BIN + ' ' +gm_fname + ' ' + solfile+'.dualdecomposition' + ' ' + 'dualdecomposition >'+logfile
		cmd_icm = MAP_BIN + ' ' +gm_fname + ' ' + solfile+'.icm' + ' ' + 'icm '+initfname+' >'+logfile
		if initLabelling is None:
			initLabelling = np.zeros(pot.shape)
		with open(initfname,'w') as f:
			f.write(" ".join([str(int(t)) for t in initLabelling[:(2*N)].reshape(N,2)[:,1].ravel().tolist()]))

		args = [('qpbo',cmd_qpbo),
				('trws',cmd_trws),
				('icm',cmd_icm)]
		pool = Pool(processes = 3)
		results = pool.map(runPAR,args)
		pool.close()
		pool.join()
		maxEnergy = -1*np.inf
		best_marginal_vec = None
		best_MAPsoln = None
		best_MAPsolver = ''
		for r in results:
			name,time_taken = r
			print name,' took ',time_taken,' seconds'
			MAPsoln = np.loadtxt(solfile+'.'+name).astype('int')
			marg_vec = util.createMarginalVector(MAPsoln,List)
			energy = np.dot(pot,marg_vec)
			print name,' Energy:',energy
			if energy>maxEnergy:
				maxEnergy = energy
				best_marginal_vec = marg_vec
				best_MAPsoln = MAPsoln
				print 'Updating best to ',name
				best_MAPsolver = name
			if logf is not None:
				logf.write('MAPsolver chosen: '+best_MAPsolver+' Energy: '+str(maxEnergy)+' Time Taken:' +str(time.time()-start)+'seconds \n')
		return best_marginal_vec,best_MAPsolver,best_MAPsoln
	else:
		#Run Map Solver
		if mapsolver=='lazyflipper' or mapsolver=='icm':
			with open(initfname,'w') as f:
				f.write(" ".join([str(int(t)) for t in initLabelling[:(2*N)].reshape(N,2)[:,1].ravel().tolist()]))
		solfile += '.'+mapsolver

		cmd = MAP_BIN + ' ' +gm_fname + ' ' + solfile + ' ' + mapsolver + ' ' + initfname +' 2'
		start = time.time()
		subprocess.check_output(cmd,shell=True,stderr = subprocess.STDOUT)
		maptime=time.time()-start

		#Get MAP solution
		MAPsoln = np.loadtxt(solfile).astype('int')

		#Get the marginal vector
		start = time.time()
		marg_vec = util.createMarginalVector(MAPsoln,List)
		margvectime = time.time()-start

		totaltime = time.time()-fxn_start_marker
		timeString = '[ConversionTime '+str(converter_time)+' MAPtime: '+str(maptime)+' MargVec Time:'+str(margvectime)+' Total: '+str(totaltime)+']'
		if logf is not None:
			logf.write(timeString+'\n')
		return marg_vec,timeString,MAPsoln
