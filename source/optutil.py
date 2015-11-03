#Optimization utilities
import numpy as np
import ipdb,sys,time,scipy
import ILPsolver
import GenericSolver
import scipy.linalg.blas as FB
import scipy.optimize as sciopt
from scipy.optimize import minimize_scalar
from scipy.sparse import *
from gurobipy import *

def setupParams(G,alg_params):
	#Setup parameters
	fxn_params = {}
	fxn_params['nEdges'] = G.nEdges
	fxn_params['nVertices'] = G.nVertices
	fxn_params['edgeIdx'] = G.edgeIdx
	fxn_params['nodeIdx'] = G.nodeIdx
	fxn_params['pot'] = G.pot
	fxn_params['epsilon']=alg_params['epsilon']
	fxn_params['edgeList']=G.Edges[:,:2].astype('int')
	fxn_params['rhos_node'] = G.rhos_node
	fxn_params['rhos_edge'] = G.rhos_edge
	fxn_params['K_c_vec']=G.K_c_vec
	f_obj = alg_params['objective_fxn']
	return fxn_params,f_obj

#Run correction step
def corrective(G,alg_params,V,alpha,bound=np.inf,logf = None):
	#Setup 
	mus = G.init_vec
	vec_len = max(mus.shape)
	fxn_params,f_obj = setupParams(G,alg_params)
	MAXITS = 100

	#Type of correction
	#if alg_params['modified_fw']:
	#	print "\t\t(MFW correction)"
	#elif alg_params['pairwise']:
	#	print "\t\t(PFW correction)"
	#else:
	#	print "\t\t(FW correction)"

	tolerance = alg_params['correction_tol']
	#print "\t\t\tIt |  Primal  | Gap    |     Step_Size   | alpha[uniform]  | AwayStep? | awayidx   | sparsity alpha"
	for it in range(MAXITS):
		start = time.time()
		val,grad = f_obj(mus,fxn_params,True)
		fxnTime = time.time()-start
		start = time.time()
		#Towards vertex
		#dot_prod_result = np.dot(VertexSet,grad)
		#toward_idx1 = np.argmin(dot_prod_result)
		#vertex = np.reshape(VertexSet[toward_idx1,:],(vec_len,))
		#Find vertex by looping
		dot_prod_result = np.inf 
		toward_idx = -1
		for idx,v in enumerate(V):
			res = np.dot(v.toarray().ravel(),grad)
			if res<dot_prod_result:
				toward_idx = idx 
				dot_prod_result = res 
		vertex = V[toward_idx].toarray().ravel()
		#Find the minimizer of the gradient among all the vertices
		if alg_params['modified_fw']:
			mus,gap,direction,step_size,extra = MFWstep(mus,grad,vertex,V,alpha,alg_params,f_obj,fxn_params)
			#Use the stricter gap for MFW 
			gap = extra['gap_PFW']
		elif alg_params['pairwise']:
			mus,gap,direction,step_size,extra = PFWstep(mus,grad,vertex,V,alpha,alg_params,f_obj,fxn_params)
		else:
			mus,gap,direction,step_size,extra = FWstep(mus,grad,vertex,V,alpha,alg_params,f_obj,fxn_params)

		bound = min(bound,-1*gap+val)
		musTime = time.time()-start
		#print "\t\t\t",it,val,gap,step_size,alpha[0],extra['away_step'],extra['away_idx'],len(np.nonzero(alpha)[0])/(1.*len(alpha))
		if np.mod(it,20)==0:
			timeStr = ('\tGradient Comp.: %.4g , musUpdateTime: %.4g')%(fxnTime,musTime)
			status = '\tIt:%d,primal: %.4g,gap: %.4g,bound: %.4g,step: %.4g'%(it,val,gap,bound,step_size)
			if logf is not None:
				logf.write(status+'\n')
				logf.write(timeStr+'\n')
		if gap<tolerance:
			break
	return mus,val,bound,extra['gap_FW']

#Localsearch with FW
def localsearch(lastVertex,mus,G,alg_params,f_obj,fxn_params,V,alpha,logf=None):
	uai_file = alg_params['toulbar2_uai_file']
	uniform = alg_params['uniform']
	mapsolver = 'icm'
	MAXITS = 30
	gap_l = [10,10,10,10,10]
	tolerance = 0.5
	val = f_obj(mus,fxn_params) 
	for it in xrange(MAXITS):
		start = time.time()
		val,grad = f_obj(mus,fxn_params,True)
		fxnTime = time.time()-start
		tmp = sys.stdout
		sys.stdout = open(os.devnull,'w')
		vertex,MAP_time_output,solution_MAP = GenericSolver.runMAP(-1*grad,G.nVertices,G.Edges,
				G.Cardinality,G.graphType,uai_file,mapsolver,lastVertex)
		savedVertex = vertex
		sys.stdout = tmp
		if np.abs(vertex-lastVertex).sum()==0:
			break
		else:
			if alg_params['pairwise']:
				mus,gap,direction,step_size,extra = PFWstep(mus,grad,vertex,V,alpha,alg_params,f_obj,fxn_params)
			elif alg_params['modified_fw']:
				mus,gap,direction,step_size,extra = MFWstep(mus,grad,vertex,V,alpha,alg_params,f_obj,fxn_params)	
			else:
				mus,gap,direction,step_size,extra = FWstep(mus,grad,vertex,V,alpha,alg_params,f_obj,fxn_params)
			gap_l[np.mod(it,5)] = gap
			status =  "\tICM It(%d): Primal: %g (%s) MAP output: %s"%(it,val,mapsolver,MAP_time_output)
			timeStr= "\ICM Fxn: %.4g, Total: %.4g Step: %.4g #Vertices:%d "%(fxnTime,time.time()-start,step_size,len(V))
			print status
			print timeStr
			if logf is not None:
				logf.write(status+'\n')
				logf.write(timeStr+'\n')
			lastVertex = savedVertex 
		if np.max(gap_l)<tolerance:
			break
		it+=1
	return mus,lastVertex,val


#Check that mus remain a convex combination of vertices 
def checkConvexCombination(mus,alpha,V,alg_params):
	return None
	if len(alpha)==0:
		return None 
	all_V = scipy.sparse.vstack(V).toarray()
	if alg_params['M_truncated'] or alg_params['M_truncated_dynamic']:
		all_V = (1-alg_params['M_eps'])*all_V+(alg_params['M_eps']*alg_params['uniform'])
	all_alpha = np.tile(np.array(alpha),(mus.shape[0],1)).T
	result = np.abs((all_V*all_alpha).sum(axis=0)-mus).sum()
	assert result<1e-8,'Not a convex combination: '+str(result)
	assert 1-np.sum(alpha)<1e-10,'Sum to 1 violated'
	return result




#extra tracks three keys 
#					1) FW_gap : step in the FW direction [M_eps]
#					2) gap_full : gap based on method over the full polytope  [M_eps]
#					3) gap_current : gap on current [regardless of M_eps]
#FW step
def FWstep(mus,grad,vertex,V,alpha,alg_params,f_obj,fxn_params):
	checkConvexCombination(mus,alpha,V,alg_params)
	extra = {}
	extra['gap_FW'] = -1
	extra['gap_current'] = -1
	extra['gap_full'] = -1
	extra['away_step'] = False
	extra['away_idx'] = -1
	
	toward_idx = findVertexInV(V, vertex,alpha)
	if alg_params['M_truncated'] or alg_params['M_truncated_dynamic']:
		direction_orig = vertex-mus
		extra['gap_FW'] = np.dot(-1*grad,direction_orig)
		extra['gap_full'] = np.dot(-1*grad,direction_orig)
		vertex = shiftedVertex(vertex,alg_params['uniform'],alg_params['M_eps'])
	direction = (vertex-mus)
	

	step_size,min_fxn_val = getStep(f_obj,fxn_params, vertex, mus)
	#Update the weights of alpha
	for idx in xrange(len(alpha)):
		alpha[idx] *= (1-step_size)
	alpha[toward_idx] += step_size
	gap = np.dot(-1*grad,direction)
	mus = mus + step_size*direction
	
	extra['gap_current'] = np.dot(-1*grad,direction)
	checkConvexCombination(mus,alpha,V,alg_params)
	clipAlphas(alpha)
	return mus,gap,direction,step_size,extra

#MFW step
def MFWstep(mus,grad,vertex,V,alpha,alg_params,f_obj,fxn_params):
	checkConvexCombination(mus,alpha,V,alg_params)
	extra = {}
	extra['gap_FW'] = -1
	extra['gap_current'] = -1
	extra['gap_full'] = -1
	extra['away_step'] = False
	extra['away_idx'] = -1
	extra['gap_PFW'] = -1
	#Away vertex
	toward_idx = findVertexInV(V, vertex,alpha) 
	vertex_away,alpha_away,away_idx = findAwayVertex(grad,V,alpha)
	max_step_size = alpha_away/float(1-alpha_away)
	extra['away_idx'] = away_idx
	#Truncated polytope
	if alg_params['M_truncated'] or alg_params['M_truncated_dynamic']:
		extra['gap_FW'] = np.dot(-1*grad,vertex-mus)
		vertex_away_shifted = shiftedVertex(vertex_away,alg_params['uniform'],alg_params['M_eps'])
		vertex_shifted = shiftedVertex(vertex,alg_params['uniform'],alg_params['M_eps'])
		condition_away = np.dot(grad,mus-vertex_away_shifted)<np.dot(grad,vertex_shifted-mus) and away_idx!=toward_idx 
		direction_pfw = vertex_shifted - vertex_away_shifted
		if condition_away:
			direction = mus-vertex_away_shifted
			extra['gap_full'] = np.dot(-1*grad,mus-vertex_away)
			step_size,min_fxn_val = getStepDir(f_obj,fxn_params, direction, mus,0,max_step_size)
			for idx in xrange(len(alpha)):
				alpha[idx]*=(1+step_size)
			alpha[away_idx] -= step_size
		else:#Do standard FW step
			direction = (vertex_shifted-mus)
			extra['gap_full'] = np.dot(-1*grad,vertex-mus)
			step_size,min_fxn_val = getStepDir(f_obj,fxn_params, direction, mus,0,1)
			for idx in xrange(len(alpha)):
				alpha[idx] *= (1-step_size)
			alpha[toward_idx] += step_size

	else:
		condition_away = np.dot(grad,mus-vertex_away)<np.dot(grad,vertex-mus) and away_idx!=toward_idx
		direction_pfw = vertex - vertex_away
		if condition_away:
			direction = mus-vertex_away
			step_size,min_fxn_val = getStepDir(f_obj,fxn_params, direction, mus,0,max_step_size)
			for idx in xrange(len(alpha)):
				alpha[idx]*=(1+step_size)
			alpha[away_idx] -= step_size
		else:#Do standard FW step
			direction = vertex-mus
			step_size,min_fxn_val = getStepDir(f_obj,fxn_params, direction, mus,0,1)
			for idx in xrange(len(alpha)):
				alpha[idx] *= (1-step_size)
			alpha[toward_idx] += step_size

	extra['away_step'] = condition_away
	gap = np.dot(-1*grad,direction)
	mus = mus + step_size*direction
	extra['gap_PFW'] = np.dot(-1*grad,direction_pfw)
	extra['gap_current'] = np.dot(-1*grad,direction)
	checkConvexCombination(mus,alpha,V,alg_params)
	clipAlphas(alpha)
	return mus,gap,direction,step_size,extra

def clipAlphas(alpha):
	for idx in xrange(len(alpha)):
		if alpha[idx]<1e-15:
			alpha[idx] = 0

#Get step size #tol = 1e-15 
def getStep(_f,params, s, x, min_range = 0, max_range = 1,tol = 1e-15):
	#return linesearch(_f,params, s-x, x, min_range, max_range,tol)
	stepSize = sciopt.fminbound(lambda a: _f(x+a*(s-x),params),min_range, max_range, xtol=tol)
	return stepSize, _f(x+stepSize*(s-x),params)

#Get step size along direction #tol = 1e-15 
def getStepDir(_f,params, direction, x, min_range = 0, max_range = 1,tol = 1e-15):
	#return linesearch(_f,params, direction, x, min_range, max_range,tol)
	assert x.shape==direction.shape," Mismatched array shapes in getStep"+str(x.shape)+" "+str(direction.shape)
	stepSize = sciopt.fminbound(lambda a: _f(x+a*direction,params),min_range, max_range, xtol=tol)
	return stepSize, _f(x+stepSize*direction,params)

def linesearch(_f,params, direction, x, min_range = 0, max_range = 1,tol = 1e-15):
	assert x.shape==direction.shape," Mismatched array shapes in getStep"+str(x.shape)+" "+str(direction.shape)
	min_fxn = lambda a: _f(x+a*direction,params)
	result = minimize_scalar(min_fxn,bounds =(min_range,max_range),method='bounded',tol = tol)
	stepSize = result.x 
	return stepSize, _f(x+stepSize*direction,params)

#Finding an away vertex 
def findAwayVertex(grad,V,alpha):
	#result = np.dot(scipy.sparse.vstack(V).toarray(),grad)
	##### old --valid_idx = np.where(np.array(alpha)>1e-15)[0]
	#valid_idx = np.where(np.array(alpha)>0)[0]
	#away_idx = valid_idx[np.argmax(result[valid_idx])]

	dot_prod_result = -1*np.inf 
	away_idx = -1
	for idx,v in enumerate(V):
		if alpha[idx]>0:
			val = np.dot(v.toarray().ravel(),grad)
			if val>dot_prod_result:
				dot_prod_result = val
				away_idx = idx 
	away_alpha = alpha[away_idx]
	return V[away_idx].toarray().ravel(), away_alpha, away_idx

#If vertex in V, return idx,else append, set alpha to 0 and return idx
def findVertexInV(V, vertex,alpha):
	#find_idx = np.where(np.sum(np.abs(scipy.sparse.vstack(V).toarray()-vertex),1)==0)[0]
	find_idx = -1
	for idx,v in enumerate(V):
		if np.abs(v-vertex).sum()==0:
			find_idx = idx 
			break 
	if find_idx==-1:
		V.append(csr_matrix(vertex))
		alpha.append(0)
		return len(alpha)-1
	else:
		return find_idx

#Shifting vertex
def shiftedVertex(v,uniform,M_eps):
	return (1-M_eps)*v+(M_eps)*uniform


#Write mus as a convex combination of the current set of vertices
def getCoeff(mus,vertices):
	model = Model("convLP")
	model.setParam( 'OutputFlag', False )
	# vertices is (# x dim)
	v = []
	#Setup model variables
	for vnum in xrange(vertices.shape[0]):
		v.append(model.addVar(lb=0,ub=1,vtype=GRB.CONTINUOUS,name="v_"+str(vnum)+"_coeff"))
	model.update()
	model.setObjective(0, GRB.MAXIMIZE)

	#Setup constraints
	#sum to 1 constraint for coefficients
	model.addConstr(LinExpr(np.ones(len(v)),v), GRB.EQUAL, 1, "sumTo1")
	#assert that for each dimension, we can express mus as a linear combination of vertices
	for dim in xrange(vertices.shape[1]):
		model.addConstr(LinExpr(vertices[:,dim],v),GRB.EQUAL,mus[dim],"dim="+str(dim))
	model.update()
	model.optimize()
	status,message,term_cond = ILPsolver.solverStatus(model)
	print "Result : ",status,message,term_cond
	try:
		coeff=np.array([a.X for a in v])
	except:
		ipdb.set_trace()
	if np.abs(np.dot(vertices.T,coeff)-mus).mean()>1e-5:
		ipdb.set_trace()
		assert False,"Conditions not satisfied"
	return coeff

