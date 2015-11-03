import ConfigParser
import argparse,ipdb,copy,glob,os,sys,time
sys.path.append('../source')
import util
import cPickle as pickle
import inferenceSpanning as inf 
import objectiveFxns as obj_fxn
from multiprocessing import Pool
from scipy.io import loadmat

infSection = "InferenceParams"
spanningSection = "SpanningTreeParams"
miscSection = "OtherParams"
#Get string to append to foldername
#Used to identify the result
def parseDocstring(config):
	alg_details = ""
	alg_details += config.get(miscSection,"pretext")
	if config.getboolean(infSection,"pairwise"):
		alg_details += "_pFW"
	if config.getboolean(infSection,"useMAP"): #Use MAP solver
		if config.get(infSection,"MAPsolver") == 'toulbar2':
			alg_details += "_MAPsolver"
		else:
			alg_details += "_approxMAPsolver"
	else: #Use Gurobi's ILP/LP solver
		if config.getboolean(infSection,"use_marginal_polytope"):
			alg_details += "_ilpMARG"
		else:
			alg_details += "_lpLOCAL"
	if config.getint(infSection,"maxSecondsPerMAP")!=-1:
		alg_details += "_TIMEOUT_"+config.get(infSection,"timeout") 
	if config.getboolean(spanningSection,"compareParams"):
		alg_details += "_PreProcess_"+config.get(spanningSection,"usePreprocess") 
		alg_details += "_FWStrategy_"+config.get(spanningSection,"FWStrategy") 
		alg_details += "_stepSize_"+config.get(spanningSection,"stepSizeComputation") 
		alg_details += "_expGrad_"+config.get(infSection,"expGrad")
	return alg_details
#Parse command line to get subfolder name
def parseSubfolder(cmdargs):
	return 'TRW'
#Setup file system
def setupFileSystem(cmdargs,config):
	docstring = parseDocstring(config)
	subfolder = parseSubfolder(cmdargs)
	mainfolder= config.get(miscSection,"rootfolder")

	outputfolder = mainfolder + '/' + cmdargs.mainfolder + '/resultsSpanning' + docstring + '/' + subfolder
	logfolder = mainfolder + '/' + cmdargs.mainfolder + '/logfilesSpanning' + docstring + '/' + subfolder

	if not os.path.exists(outputfolder):
		os.makedirs(outputfolder)
	if not os.path.exists(logfolder):
		os.makedirs(logfolder)
	return logfolder,outputfolder 

#Setup alg_params (for inference in inner loop) and SpanningTreeParams
checkBool = lambda x: x=='True' or x=='False'
toBool    = lambda x: x=='True'
def parseAlgParams(cmdargs,config):
	alg_params = {}
	for (key,val) in config.items(infSection):
		alg_params[key]=val
		if checkBool(val):
			alg_params[key] = toBool(val)
		if val.replace("-","").isdigit():
			alg_params[key] = int(val)
	#Set tolerance as float
	alg_params['tol'] = config.getfloat(infSection,"tol")
	alg_params['M_eps'] = config.getfloat(infSection,"M_eps")
	alg_params['epsilon'] = cmdargs.epsilon
	alg_params['graphType'] = cmdargs.graphType
	alg_params['initRhos'] = config.getboolean(miscSection,"initRhos")
	alg_params['rhosFolder'] = config.get(miscSection,"rootfolder")+'/'+config.get(miscSection,"rhosFolder")
	return alg_params
def parseSpanningTreeParams(cmdargs,config):
	spanning_tree_params = {}
	for (key,val) in config.items(spanningSection):
		spanning_tree_params [key]=val
		if checkBool(val):
			spanning_tree_params[key] = toBool(val)
		if val.replace("-","").isdigit():
			spanning_tree_params[key] = int(val)
	return spanning_tree_params

#Run Marginal Inference
def runInference(args):
	pid = os.getpid()
	#Unpack args 
	matf,gfile,alg_params,spanning_tree_params = args
	print pid,"| Running on ",matf, " graphFile: ",gfile
	#Setup objective function
	alg_params['objective_fxn'] = obj_fxn.TRW
	N_n,Nodes_n,Edges_n,Cardinality_n,mode_n = util.loadUAIfromMAT(matf)
	start_time = time.time()
	G = util.Graph(N=N_n,graphFile = gfile, type=alg_params['graphType'],
			Edges=Edges_n,Nodes=Nodes_n,Cardinality=Cardinality_n,
			mode='UAI',weighted=True)
	spanning_tree_params['alg_params']=alg_params
	inf.optSpanningTree(G,spanning_tree_params)
	print pid,"| Completed  ",matf," time elapsed: "
	time_taken = time.time()-start_time
	return (pid,time_taken,os.path.basename(matf).replace('.mat',''))

#Run experiment
def runExpt(cmdargs):
	#Get configuration file
	config = ConfigParser.ConfigParser()
	config.optionxform = str 
	config.read('./config/default.cfg')
	config.read(cmdargs.config_fname)
	#Strip settings from config file
	alg_params = parseAlgParams(cmdargs,config)
	spanning_tree_params = parseSpanningTreeParams(cmdargs,config)
	#Special cases for running in parallel
	if not (alg_params['useMAP'] and alg_params['MAPsolver']=='toulbar2'):
		print 'WARNING: Can only run in parallel with toulbar2'
		cmdargs.parallel = False
	if cmdargs.testcase != '-1':
		cmdargs.parallel = False
	if cmdargs.parallel:
		alg_params['quiet']=True
		spanning_tree_params['quiet'] = True 
	#Setup file system
	logfolder,outputfolder = setupFileSystem(cmdargs,config)
	############ Setup Run for Relevant Instances  ###########
	runs= []
	print config.get(miscSection,"rootfolder")+'/'+cmdargs.mainfolder+'/uai_mat/*.mat'
	for matfile in glob.glob(config.get(miscSection,"rootfolder")+'/'+cmdargs.mainfolder+'/uai_mat/*.mat'):
		print "doing ",matfile
		a_params = copy.deepcopy(alg_params)
		s_params = copy.deepcopy(spanning_tree_params)
		basename = os.path.basename(matfile).replace('.mat','')
		if cmdargs.testcase != '-1' and basename != cmdargs.testcase:
			if not cmdargs.quiet:
				#print "Skipping ",basename
				pass
			continue
		#Setup logfiles
		a_params['log_file_name'] = logfolder + '/' + basename + '-textlog.txt'
		a_params['grd_log_file'] = logfolder + '/' + basename + '-grb.txt'
		a_params['toulbar2_uai_file'] = logfolder + '/' + basename + '-MAP.uai'
		a_params['mat_file_name'] = outputfolder + '/' + basename + '.mat'
		gfile = logfolder + '/' + basename + '-graph.grph'
		s_params['log_file_name'] = logfolder + '/' + basename + '-txtLogSpanning.txt'
		s_params['mat_file_name'] = outputfolder + '/' + basename + '-spanning.mat'
		#Print Details
		if not cmdargs.quiet:
			print basename,"| Results in: ",outputfolder," | Logfiles in: ",logfolder," | gfile : ",gfile
			print "---- Inference Params ----"
			for k in a_params:
				print k,"->",a_params[k]
			print "---- ----- ----"
			print "---- Spanning Tree Params ----"
			for k in s_params:
				print k,"->",s_params[k]
			print "---- ----- ----"
		runs.append((matfile,gfile,a_params,s_params))
	#Start inference
	start_time = time.time()
	results = map(runInference,runs)
	print "--------- Complete. Runtime Stats -------- "
	print "Total Time Taken: ",(time.time()-start_time)/60," minutes"
	for (pid,time_taken,bname) in results:
		print "File: ",bname," PID: ",pid," Time Taken: ",time_taken

#Main Function	
if __name__=='__main__':
	#Parse arguments : mainfolder, tc, epsilon, graphType, configFile
	parser = argparse.ArgumentParser(description = 'Inference with the TRW objective',
			epilog='Author: Rahul G. Krishnan (rahul@cs.nyu.edu)')
	parser.add_argument('mainfolder',help = 'Mainfolder that has been created using setup-scripts')
	parser.add_argument('testcase',help = 'Basename (name w/o ext). -1 runs all test cases in directory.')
	parser.add_argument('epsilon',type=float,help = 'Value of epsilon. 0 for no approximation')
	parser.add_argument('graphType',choices = ['dir','undir'],default='undir',help = 'Type of graph. <dir> uses the \
			conditional entropy and <undir> uses mutual information.')
	parser.add_argument('config_fname',help = 'Location of configuration file. Contains other params')
	parser.add_argument('-parallel',action = 'store_true',default = False,help = 'Flag to run in parallel (Does not work)')
	parser.add_argument('-quiet',action = 'store_true',default = False,help = 'Flag to run queitly')
	cmdargs = parser.parse_args()
	runExpt(cmdargs)
