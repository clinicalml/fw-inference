#File to run inference with Frank Wolfe
from __future__ import division
import sys,time,datetime,os
import numpy as np
from util import *
import objectiveFxns as obj_fxn
import optutil,pickle
from scipy.sparse import *

#Solvers for MAP inference
import MAPsolver
import ILPsolver
import GenericSolver

#Setup parameters
def setupParams(G,alg_params):
	#Setup arguments to functions : Do not modify
	fxn_params = {}
	fxn_params['nEdges'] = G.nEdges
	fxn_params['nVertices'] = G.nVertices
	fxn_params['Cardinality'] = G.Cardinality
	fxn_params['edgeIdx'] = G.edgeIdx
	fxn_params['nodeIdx'] = G.nodeIdx
	fxn_params['pot'] = G.pot
	fxn_params['epsilon']=alg_params['epsilon']
	fxn_params['edgeList'] = G.Edges[:,:2].astype('int')
	fxn_params['K_c_vec']=G.K_c_vec
	fxn_params['rhos_node'] = G.rhos_node
	fxn_params['rhos_edge'] = G.rhos_edge
	#Bookkeeping
	if 'toulbar2_uai_file' not in alg_params:
		alg_params['toulbar2_uai_file'] = './temp.uai'
	if 'maxSecondsPerMAP' not in alg_params:
		alg_params['maxSecondsPerMAP'] = -1

	#Setup objective function
	f_obj = alg_params['objective_fxn']

	#Logging
	if 'appendToFiles' in alg_params:
		logf = open(alg_params['log_file_name'],'a')
	else:
		logf = open(alg_params['log_file_name'],'w')
	logf.write('Starting Marginal Inference\n'+str(datetime.datetime.now())+'\n')
	matf = alg_params['mat_file_name']
	uniform = np.zeros(G.init_vec.shape)
	uniform[:G.nVertices*2]=0.5
	uniform[G.nVertices*2:]=0.25
	alg_params['uniform'] = uniform
	return logf,matf,fxn_params,f_obj


#Run marginal inference with FW
def runMarginalFW(G,alg_params,model=None,V = [],alpha = [],stepSizeComputation = False,rho_vec =-1):
	quiet = alg_params['quiet'] 
	if stepSizeComputation:
		quiet = True
		assert type(rho_vec)!=int,'Requires rho_vec to be a vector'
		G.rhos_edge = rho_vec
		G.computeNodeRhos()
	######### Setup parameters $$$$$$$$$$
	stdout_initial = -1
	dev_null_f = open(os.devnull,'w')
	if quiet:
		stdout_initial = sys.stdout
		sys.stdout = dev_null_f
	logf,matf,fxn_params,f_obj = setupParams(G,alg_params)
	#print "\t----- Starting Marginal Inference with FW ------ \t"

	#Initial settings
	mus = G.init_vec
	MAX_STEPS = alg_params['max_steps']
	smallest_marginal = np.Inf
	if len(alpha)==0:
		V.append(csr_matrix(mus))
		alpha.append(1)

	#Track statistics
	statistics= setupStatistics(MAX_STEPS,mus.shape,G.nVertices)
	statistics['alpha'] = alpha

	#Extract gurobi model 
	if not alg_params['useMAP'] and model is None:
		model = ILPsolver.defineModel(G,alg_params)

	if alg_params['useMAP'] and alg_params['MAPsolver']!='toulbar2':
		len_gap = 5
	else:
		len_gap = 1
	objWarning =0
	gap_l = [10]*len_gap

	if alg_params['M_truncated_dynamic'] or alg_params['M_truncated']:
		#Additional stats to track 
		#Updated at every iteration 
		statistics['gap_FW'] = []
		statistics['gap_full'] = []
		#Updated when epsilon modified 
		statistics['eps_val'] = []
		statistics['primal_push'] = []
		statistics['iterate_push'] = []
		statistics['marker'] = []
		statistics['eps_val'] =[alg_params['M_eps']]
		#print "------------- M_eps with eps = ",alg_params['M_eps'],' -----------------'

	if alg_params['PreCorrection']:
		alg_params['correction_tol'] = 0.5
		G.init_vec = mus
		mus,val,bound,correctionGap = optutil.corrective(G,alg_params,V,alpha,np.inf,logf)

	#For all the steps
	for it in xrange(MAX_STEPS):
		assert len(alpha)==len(V),'mismatch in alpha and V'
		start_time = time.time()
		val,grad = f_obj(mus,fxn_params,True)
		#################   Running MAP Inference ################
		if not alg_params['useMAP']:
			model = ILPsolver.updateObjective(grad,alg_params,model)
		if alg_params['useMAP']:
			map_potentials = -1*grad
			if alg_params['MAPsolver']=='toulbar2':
				vertex,toulbar2_output,solution_MAP = MAPsolver.runMAP(map_potentials,G.nVertices,G.Edges,
					G.Cardinality,G.graphType,alg_params['toulbar2_uai_file'],alg_params['maxSecondsPerMAP'])
			elif alg_params['MAPsolver']=='MPLP':
				vertex,MPLP_output,solution_MAP = MPLPsolver.runMAP(map_potentials,G.nVertices,G.Edges,
					G.Cardinality,G.graphType,alg_params['toulbar2_uai_file'],alg_params['maxSecondsPerMAP'])
			else:#Use MAP solvers from openGM
				if it<2:
					vInit = None
				else:
					vInit = V[-1].toarray()[0]
				vertex,MAP_output,solution_MAP = GenericSolver.runMAP(map_potentials,G.nVertices,G.Edges,
						G.Cardinality,G.graphType,alg_params['toulbar2_uai_file'],'all',vInit,logf)
		else:
			vertex = ILPsolver.optimize(model)
		################   Code to track usage and save statistics ##################
		if alg_params['use_marginal_polytope']:
			assert np.sum(vertex-vertex.astype(int))<np.exp(-10),"Vertex not integer. Investigate : "+str(vertex)
		###############   Update marginals ###############
		#ipdb.set_trace()

		#Check if epsilon needs to be updated 
		if alg_params['M_truncated_dynamic']:
			g_u0 = np.dot(-1*grad,alg_params['uniform']-mus)
			g_k = np.dot(-1*grad,vertex-mus)
			#If the gap is negative use the correctionGap
			if g_k<0:
				g_k = correctionGap 
			if g_u0==0:
				new_eps = np.inf
			elif g_u0<0:
				new_eps = g_k/(-4*g_u0)
			else:
				new_eps = alg_params['M_eps']
			#Halve epsilon if the gap is *still* negative (shouldn't happen)
			#and if below precision -> set to 0 
			if new_eps < 0:
				new_eps = alg_params['M_eps']/2.
				if new_eps<1e-15:
					new_eps = 0
			if new_eps<alg_params['M_eps']:
				#Modified version to ensuring halving
				new_eps = min(new_eps,alg_params['M_eps']/2.)
				print "************ UPDATING DYN. EPSILON TO ",new_eps," *******************"
				old_eps = alg_params['M_eps']
				for i in xrange(1,len(alpha)):
					alpha[i] = alpha[i]*((1-old_eps)/(1-new_eps))
				alpha[0] = alpha[0] - (1-alpha[0])*(((1-old_eps)/(1-new_eps))*new_eps - old_eps)
				alg_params['M_eps'] = new_eps 
				#Check what an away step would look like 
				step_size,min_fxn_val = optutil.getStepDir(f_obj,fxn_params, mus-alg_params['uniform'],mus,0,alpha[0]/(1-alpha[0]))
				statistics['primal_push'].append(min_fxn_val)
				statistics['iterate_push'].append(mus+step_size*(mus-alg_params['uniform']))
				statistics['eps_val'].append(new_eps)
				statistics['marker'].append(it)
				alg_params['correction_tol'] = 0.5
				G.init_vec = mus
				mus,val,bound,correctionGap = optutil.corrective(G,alg_params,V,alpha,np.inf,logf)

		if alg_params['pairwise']:
			mus,gap,direction,step_size,extra = optutil.PFWstep(mus,grad,vertex,V,alpha,alg_params,f_obj,fxn_params)
		else:
			mus,gap,direction,step_size,extra = optutil.FWstep(mus,grad,vertex,V,alpha,alg_params,f_obj,fxn_params)
			#if is eps -> uses extra['gap_FW'] otherwise uses gap wchich is the FW gap anyways

		#Fully Corrective Variant
		if np.mod(it,alg_params['correctionFreq'])==0 and alg_params['doCorrection']:
			G.init_vec = mus
			if gap < 1:
				alg_params['correction_tol'] = 0.05
			else:
				alg_params['correction_tol'] = 0.5
			mus,val,bound,correctionGap = optutil.corrective(G,alg_params,V,alpha,np.inf,logf)
		#Local Search
		if alg_params['doICM']:
			mus,vertex_last,val = optutil.localsearch(vertex,mus,G,alg_params,f_obj,
				fxn_params,V,alpha,logf)

		#Track statistics
		entropy = np.sum(-1*np.log(mus)*mus)
		val_do_not_use,grad_final = f_obj(mus,fxn_params,True)
		grad_norm = np.max(grad_final)
		dir_norm = np.linalg.norm(direction)		
		if np.min(mus)<smallest_marginal:
			smallest_marginal = np.min(mus)
		updateStatistics(statistics,it,time.time()-start_time,len(V),mus,val,gap,step_size,entropy,grad_norm,dir_norm,np.min(mus))

		#Also track statisitcs for 
		if alg_params['M_truncated'] or alg_params['M_truncated_dynamic']:
			statistics['gap_FW'].append(extra['gap_FW'])
			statistics['gap_full'].append(extra['gap_full'])
			writeStatistics(statistics,it,matf,alsoWrite = ['gap_FW','gap_FW'])
			print 'Gap FW: ',extra['gap_FW']
		else:
			writeStatistics(statistics,it,matf)

		#Error : Objective not decreasing
		if it>2 and statistics['Obj_Val'][it]>statistics['Obj_Val'][it-1] and np.abs(statistics['Obj_Val'][it]-statistics['Obj_Val'][it-1])>1e-7:
			logf.write('####### WARNING. OBJECTIVE NOT DECREASING ######\n')
			logf.write('Step Size: '+str(step_size)+' Diff: '+str(np.abs(statistics['Obj_Val'][it]-statistics['Obj_Val'][it-1]))+'\n')
			print "WARNING: ",'Step Size: '+str(step_size)+'Obj: '+str(statistics['Obj_Val'][it])+' Diff: '+str(np.abs(statistics['Obj_Val'][it]-statistics['Obj_Val'][it-1]))+'\n'
			objWarning +=1
			if objWarning>3:
				logf.write('#### EXITING due to increasing objective #####\n')
				break
			assert False,'Objective should be decreasing'

		################## Print/Write to console ##################
		entropy_mus = np.sum(obj_fxn.computeEntropy(mus[1:np.sum(G.Cardinality)]))
		print "\n"
		status_1 = 	'It: %d, Primal: %.8g, Gap: %.8g, Bound: %.5g ' % (it,val,gap,statistics['Bound'][-1])
		status_2 = 	'Time(s): %.3g, StepSize: %.8g  ' % (statistics['Runtime'][-1],step_size)
		status_3 = 	'Entropy(mus)= %.3g, Norm (p) : %.3g, Norm (g) : %.3g '% (entropy_mus,dir_norm,grad_norm)
		status_4 =  'Min mu (overall) = %.5g, Min mu (current) = %.5g Sum(alphas) = %.3g' %(smallest_marginal,np.min(mus),np.sum(alpha))
		status = status_1+status_2+status_3+status_4
		print status
		logf.write(status+'\n')

		##################  Check stopping criterion ################
		gap_l[np.mod(it,len_gap)] = gap
		#M_eps variants have different stopping criterion 
		if alg_params['M_truncated'] or alg_params['M_truncated_dynamic']:
			#All variants break if global gap is less than tolerance 
			if extra['gap_FW']<alg_params['tol']:
				break 
			#Modify eps 
			if alg_params['M_truncated'] and alg_params['M_eps_iterations']>1 and gap < alg_params['tol']: #Manually modify eps
				old_eps = alg_params['M_eps']
				new_eps = alg_params['M_eps']/2.0
				print "************ UPDATING EPSILON TO ",new_eps," *******************"
				for i in xrange(1,len(alpha)):
					alpha[i] = alpha[i]*((1-old_eps)/(1-new_eps))
				alpha[0] = alpha[0] - (1-alpha[0])*(((1-old_eps)/(1-new_eps))*new_eps - old_eps)
				alg_params['M_eps'] = new_eps 
				#Check what an away step would look like 
				step_size,min_fxn_val = optutil.getStepDir(f_obj,fxn_params, mus-alg_params['uniform'],mus,0,alpha[0]/(1-alpha[0]))
				statistics['primal_push'].append(min_fxn_val)
				statistics['iterate_push'].append(mus+step_size*(mus-alg_params['uniform']))
				statistics['eps_val'].append(new_eps)
				statistics['marker'].append(it)
				alg_params['M_eps_iterations'] = alg_params['M_eps_iterations'] - 1
		else:
			if np.max(gap_l)<alg_params['tol']:
				print "Duality Gap condition reached: ",np.mean(gap_l)
				break

	print "\n----- Done Marginal Inference ------ \n"
	################ Cleanup Code ###############
	if alg_params['M_truncated'] or alg_params['M_truncated_dynamic']:
		old_eps = alg_params['M_eps']
		new_eps = alg_params['M_eps']/2.0
		for i in xrange(1,len(alpha)):
			alpha[i] = alpha[i]*((1-old_eps)/(1-new_eps))
		alpha[0] = alpha[0] - (1-alpha[0])*(((1-old_eps)/(1-new_eps))*new_eps - old_eps)
		alg_params['M_eps'] = new_eps 
		#Check what an away step would look like 
		#print alpha[0]
		step_size,min_fxn_val = optutil.getStepDir(f_obj,fxn_params, mus-alg_params['uniform'],mus,0,alpha[0]/(1-alpha[0]))
		statistics['primal_push'].append(min_fxn_val)
		statistics['iterate_push'].append(mus+step_size*(mus-alg_params['uniform']))
		statistics['eps_val'].append(new_eps)
		statistics['marker'].append(it)

		mat = writeStatistics(statistics,it,matf,returnMAT = True,alsoWrite = ['gap_FW','gap_full','primal_push','eps_val','marker'])
		mat['iterate_push'] = np.vstack(statistics['iterate_push'])
	else:
		mat = writeStatistics(statistics,it,matf,returnMAT = True)

	print "LogZ: ",val*-1+gap
	print "Marginals: ",
	#for m in statistics['IterSet'][it,:np.sum(G.Cardinality)].tolist():
	#	print ('%.4f')%(m),
	n_m = statistics['IterSet'][it,:np.sum(G.Cardinality)]
	print n_m.reshape(G.nVertices,2)
	e_m = statistics['IterSet'][it,np.sum(G.Cardinality):]
	print e_m.reshape(G.nEdges,4)

	dev_null_f.close()
	logf.close()

	if quiet:
		sys.stdout=stdout_initial
	if stepSizeComputation:
		return val*-1+gap
	else:
		return mat

#Setup, update and write statistics
def setupStatistics(MAX_STEPS,shape,N):
	statistics = {}
	#dynamic
	statistics['UniqVert'] = [] 
	statistics['IterSet'] = np.zeros((MAX_STEPS,max(shape)))
	statistics['Obj_Val'] = [] 
	statistics['Dual_Gap']= [] 
	statistics['Bound']= [-1*np.inf] 
	statistics['Runtime'] = []
	statistics['Step_Size']= []
	statistics['Entropy']= []
	statistics['Direction_Norm']= []
	statistics['Gradient_Norm']= []
	statistics['Smallest_Marginal']= []
	#static 
	statistics['N'] = N
	return statistics 

def updateStatistics(statistics,it,runtime,uniq_vert,mus,val,gap,step_size,entropy,grad_norm,dir_norm,smallest_marginal,extra = None):
	statistics['UniqVert'].append(uniq_vert)
	statistics['IterSet'][it,:] = mus
	statistics['Obj_Val'].append(val)
	statistics['Dual_Gap'].append(gap)
	statistics['Bound'].append(min(statistics['Bound'][-1],-1*val+gap))
	statistics['Runtime'].append(runtime)
	statistics['Step_Size'].append(step_size)
	statistics['Entropy'].append(entropy)
	statistics['Direction_Norm'].append(dir_norm)
	statistics['Gradient_Norm'].append(grad_norm)
	statistics['Smallest_Marginal'].append(smallest_marginal)
	if extra is not None:
		for k in extra:
			statistics[k].append(extra[k])

def writeStatistics(statistics,it,matf,returnMAT=False,alsoWrite=None):
	mat = {}
	mat['UniqueVert'] = np.array(statistics['UniqVert'])
	mat['alpha']= np.array(statistics['alpha'])
	mat['N'] =  statistics['N']
	mat['IterSet'] = statistics['IterSet'][:(it+1),:]
	mat['Obj_Val'] = np.array(statistics['Obj_Val'])
	mat['Dual_Gap'] = np.array(statistics['Dual_Gap'])
	mat['Bound'] = np.array(statistics['Bound'])
	mat['Runtime'] = np.array(statistics['Runtime'])
	mat['Step_Size'] = np.array(statistics['Step_Size'])
	mat['Entropy'] = np.array(statistics['Entropy'])
	mat['Direction_Norm'] = np.array(statistics['Direction_Norm'])
	mat['Gradient_Norm'] = np.array(statistics['Gradient_Norm'])
	mat['Smallest_Marginal'] = np.array(statistics['Smallest_Marginal'])
	if alsoWrite is not None:
		for k in alsoWrite:
			mat[k] = np.array(statistics[k])
	savemat(matf, mat)
	if returnMAT:
		return mat 
	else:
		return None

#Collect and return the unique vertices found thus far
def numUniqueVerices(VertexSet,returnUnique = False):
	VertexSet_new = np.unique([tuple(row) for row in VertexSet])
	vertex_ctr = VertexSet_new.shape[0]
	if returnUnique:
		return vertex_ctr,VertexSet_new
	else:
		return vertex_ctr


if __name__ == '__main__':
	mainfolder = '/data/ml2/rahul/FrankWolfeMarginalInf/Debug3Node'
	alg_params = {}
	alg_params['use_marginal_polytope'] = True
	alg_params['use_cycle_inequalities'] = False
	alg_params['useAway'] = True
	alg_params['tol'] = 0.5
	alg_params['max_steps'] = 20
	alg_params['usePARTAN'] = False
	alg_params['warm_start'] = True
	alg_params['useMAP'] = False
	alg_params['maxSecondsPerMAP'] = -1
	alg_params['grb_log_to_console'] = 0
	alg_params['epsilon'] = 0
	inputfolder = mainfolder+'/uai_mat'
	exactfolder = mainfolder +'/mat_out'
	outputfolder= mainfolder+'/results'
	logfolder   = mainfolder+'/logfiles'
	createIfAbsent([outputfolder,logfolder])

	alg_params['objective_fxn'] = obj_fxn.COND
	subfolder = "COND"

	basename = 'N3Tr1'
	logfolder_run = logfolder+'/'+subfolder
	outputfolder_run = outputfolder+'/'+subfolder
	createIfAbsent([outputfolder_run,logfolder_run])

	logfilename = logfolder_run+'/log-'+basename+'-textlog.txt'
	alg_params['log_file_name'] = logfilename
	#Gurobi log file
	grb_log_file = logfolder_run+'/log-'+basename+'-grb.txt'
	alg_params['grb_log_file'] = grb_log_file
	#toulbar2 UAI file
	toulbar2_uai_file = logfolder_run+'/'+basename+'-MAP.uai'
	alg_params['toulbar2_uai_file'] = toulbar2_uai_file
	#MATLAB file with results
	matfilename = outputfolder_run+'/'+basename+'.mat'
	alg_params['mat_file_name'] = matfilename

	N_n,Nodes_n,Edges_n,Cardinality_n,mode_n = loadUAIfromMAT(mainfolder+'/uai_mat/'+basename+'.mat')
	#Modify edges to also add edges between latent variables
	G = Graph(N=N_n,Edges=Edges_n,Nodes=Nodes_n,Cardinality=Cardinality_n,
		mode=mode_n,weighted=True,type='dir')
	final_results = runMarginalFW(G,alg_params)

	G = Graph(N=N_n,Edges=Edges_n,Nodes=Nodes_n,Cardinality=Cardinality_n,
		mode=mode_n,weighted=True,type='undir')
	alg_params['objective_fxn']=obj_fxn.TRW_OPT
	final_results = runMarginalFW(G,alg_params)

