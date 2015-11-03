#Code to perform optimization over the spanning tree polytope
from __future__ import division
import inferenceFW as inf
import numpy as np
import scipy.optimize as sciopt
import os,sys,time,copy,ipdb,util,scipy,optutil,ILPsolver
import cPickle as pickle
from scipy.io import savemat 
import spanningUtilities as spanUtil
import objectiveFxns as obj_fxn

#optimize spanning tree polytope
#run marginal inference with FW
def optSpanningTree(G,spanning_tree_params):
	quiet = spanning_tree_params['quiet']
	dev_null_f = open(os.devnull,'w')
	if quiet:
		stdout_initial = sys.stdout
		sys.stdout = dev_null_f
	print "\n----- Starting Marginal Inference - Optimizing Spanning Tree Polytope ------ \n"

	#Dealing with parameters
	alg_params = copy.deepcopy(spanning_tree_params['alg_params'])
	spanning_tree_iter = spanning_tree_params['spanning_tree_iter']
	matf = spanning_tree_params['mat_file_name']
	logfname = spanning_tree_params['log_file_name']

	assert 'usePreprocess' in spanning_tree_params,'usePreprocess not set'
	assert 'FWStrategy' in spanning_tree_params and 'Kval' in spanning_tree_params,'FWStrategy/Kval not set'
	assert 'stepSizeComputation' in spanning_tree_params,'stepSizeComputation not set'

	model = None
	if not alg_params['useMAP']:
		model = ILPsolver.defineModel(G,alg_params)

	#Keep aside a different log/mat file for every iteration of optimizing over the spanning tree
	alg_matf = alg_params['mat_file_name'].replace('.mat','')
	alg_logf = alg_params['log_file_name'].replace('.txt','')

	logf = open(logfname,'w',0)

	#Tracking unique vertices of the marginal polytope
	all_data = []
	vec_len = max(G.init_vec.shape)

	V = []
	alpha = []
	prevUb = np.inf

	########### Running iterations of Frank-Wolfe #################
	for it in xrange(spanning_tree_iter):
		print "\nIteration ",(it+1)

		#Update the name of matlab/log file for every iteration
		alg_params['mat_file_name'] = alg_matf+'-sp'+str(it)+'.mat'
		alg_params['log_file_name'] = alg_logf+'-sp'+str(it)+'.txt'

	################# Strategies for marginal inference with FW #############################
		if spanning_tree_params['FWStrategy'] == 'runK' and it>0:
			#Truncate the number of ILP calls
			alg_params['max_steps'] = spanning_tree_params['Kval']
		elif spanning_tree_params['FWStrategy'] == 'uniform':
			pass
		else:
			pass

		if spanning_tree_params['usePreprocess'] and it>0:
			alg_params['PreCorrection'] = True 
		else:
			alg_params['PreCorrection'] = False 
		
		mat = inf.runMarginalFW(G,alg_params,model,V,alpha)
		if len(alpha) != len(mat['alpha']):
			alpha = mat['alpha']
		val = mat['Obj_Val'][-1]
		mus = mat['IterSet'][-1,:]
		#postprocess
		start_time = time.time()
		mat['Runtime'][-1] = mat['Runtime'][-1]+(time.time()-start_time)
	########################   Update Status ####################################

		#print Status
		status = 'Stats: #Itns: %d, Primal Obj: %.6f,\
				Final Gap: %.6f, Avg Runtime(seconds): %.3f,\
				Last Step Size: %f, #Unique Vertices : %d' % \
				(mat['Obj_Val'].shape[0],mat['Obj_Val'][-1],
				mat['Dual_Gap'][-1],np.mean(mat['Runtime']),
				mat['Step_Size'][-1],len(V))
		#print status
		if quiet:
			stdout_initial.write(str(os.getpid())+" "+str(status)+"\n")

		logf.write(status+'\n')

	#####################  Updating rho vectors ###############################
		start_time = time.time()
		G.init_vec  = mus
		spanning_tree_polytope_vertex,grad = spanUtil.computeMST(G,mus)
		direction = np.reshape(spanning_tree_polytope_vertex,(spanning_tree_polytope_vertex.shape[0],1))-G.rhos_edge
		rhos = G.rhos_edge
		status = "Computing directed spanning tree took "+str(time.time()-start_time)+' seconds'
		#print status
		logf.write(status+'\n')
		#Strategies for computing the step size
		if spanning_tree_params['stepSizeComputation'] == 'linesearch':
			G_copy = copy.deepcopy(G)
			alpha_rho = sciopt.fminbound(lambda a:inf.runMarginalFW(G_copy,alg_params,model,V,alpha,stepSizeComputation=True,rho_vec = rhos+a*direction),0,1,xtol=0.0005)#,disp=3)
			print "Step Size (Vertices) :",alpha_rho
		elif spanning_tree_params['stepSizeComputation'] == 'standard':
			alpha_rho = float(2)/(it+3)
			print "Step Size (Standard) :",alpha_rho
		else:
			assert False,'Invalid stepSizeComputation: '+spanning_tree_params['stepSizeComputation']
		#Update
		G.rhos_edge = G.rhos_edge+alpha_rho*(direction)
		#Update rhos_node and K_C
		G.computeNodeRhos()
		print "Edge Appearance\n",G.rhos_edge," \nDirection\n",spanning_tree_polytope_vertex
		#print "LogZ estimate: ",inf.runMarginalFW(G_copy,alg_params,model,V,alpha,stepSizeComputation=True,rho_vec = rhos+alpha_rho*direction)
	############################# Collect Data and Save ###############################
		#Collect data on this iteration
		data = {}
		data['fw_result'] = mat
		data['rhos'] = G.rhos_edge
		data['alpha_rho'] = alpha_rho
		data['timeTaken'] = time.time()-start_time + np.sum(mat['Runtime'])
		data['dualityGap'] = -1*np.dot(grad,direction) 
		print "Rho Gap: ",data['dualityGap']

		#Postprocessing using the vertices of the marginal polytope
		if spanning_tree_params['usePreprocess'] and it<spanning_tree_iter-1:
			start_time = time.time()
			mus,val,prevUb,gapFW = optutil.corrective(G,alg_params,V,alpha,bound=prevUb)
			G.init_vec = mus
			data['timeTaken'] += (time.time()-start_time)

		#Append stats at the very end
		all_data.append(data)
		all_data_mat = {}
		all_data_mat['final_mus'] = mus
		all_data_mat['final_logz'] = val
		all_data_mat['rho_data'] = all_data
		all_data_mat['MARG_vertices'] = scipy.sparse.vstack(V)
		savemat(matf,all_data_mat)

	print "\n----- Done Marginal Inference ------ \n"
	logf.close()
	if quiet:
		sys.stdout=stdout_initial
	dev_null_f.close()

