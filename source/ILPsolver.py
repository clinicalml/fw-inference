#Use gurobi to define and ILP solver
from gurobipy import *
import os,ipdb,sys
import numpy as np

#Update objective
def updateObjective(obj,alg_params,model):
	var = model.getVars()
	o = LinExpr([float(z) for z in obj.tolist()], var)
	if not alg_params['warm_start']:
		model.reset()
	model.setObjective(o, GRB.MINIMIZE)
	model.update()
	return model

#Optimize and return solution
def optimize(model):
	dev_null_f = open(os.devnull,'w')
	stdout_initial = sys.stdout
	sys.stdout = dev_null_f
	model.optimize()
	sys.stdout=stdout_initial
	dev_null_f.close()
	return np.array([v.X for v in model.getVars()])

#Define constraints for the local consistency (or marginal) polytope for pairwise MRF
def defineModel(G,alg_params):
	#Number of variables in optimization problem
	nVars = len(G.pot)
	model = Model()
	EdgeList = G.Edges[:,:2].astype(int)

	#Collect list of variables
	var = []
	for vi in xrange(G.nVertices):
		for i in xrange(G.Cardinality[vi]):
			v_name = "|v"+str(vi)+'='+str(i)+"|"
			if alg_params['use_marginal_polytope']:
				var.append(model.addVar(vtype=GRB.BINARY,name=v_name))
			else:
				var.append(model.addVar(vtype=GRB.CONTINUOUS,name=v_name))

	for ei in xrange(G.nEdges):
		v1 = EdgeList[ei,0]
		v2 = EdgeList[ei,1]
		for i in xrange(G.Cardinality[v1]):
			for j in xrange(G.Cardinality[v2]):
				e_name = "|e"+str(ei)+' <v'+str(v1)+'='+str(i)+'-v'+str(v2)+'='+str(j)+'>|'
				if alg_params['use_marginal_polytope']:
					var.append(model.addVar(vtype=GRB.BINARY,name=e_name))
				else:
					var.append(model.addVar(vtype=GRB.CONTINUOUS,name=e_name))
	assert nVars==len(var),"Mismatched variables and expected variables: "+str(len(var))+" vs "+str(nVars)
	model.update()

	#Set upper bound and lower bounds
	for i in xrange(nVars):
		var[i].setAttr("lb", float(0))
		var[i].setAttr("ub", float(1))
	#Set objective
	o = LinExpr([float(z) for z in G.init_vec.tolist()], var)
	#Remove old objective and set new one
	model.reset()
	model.setObjective(o, GRB.MINIMIZE)
	model.update()

	#Set constraints defining local polytope
	#Sum to 1 constraints
	for vi in xrange(G.nodeIdx.shape[0]):
		name = "|Sum to 1: Node "+str(vi)+"- idx_sum from"+str(G.nodeIdx[vi,0])+"-"+str(G.nodeIdx[vi,1])+"|"
		model.addConstr(quicksum(var[G.nodeIdx[vi,0]:G.nodeIdx[vi,1]]),GRB.EQUAL,1,name)

	for ei in xrange(G.edgeIdx.shape[0]):
		v1 = EdgeList[ei,0]
		v2 = EdgeList[ei,1]
		name = "|Sum to 1: Edge "+str(ei)+": "+str(v1)+"-"+str(v2)+"- idx_sum from"+str(G.edgeIdx[ei,0])+"-"+str(G.edgeIdx[ei,1])+"|"
		model.addConstr(quicksum(var[G.edgeIdx[ei,0]:G.edgeIdx[ei,1]]),GRB.EQUAL,1,name)

	#Pairwise consistency constraints
	#For every edge marginalize over value of unary potential
	for ei in xrange(G.edgeIdx.shape[0]):
		v1 = EdgeList[ei,0]
		v2 = EdgeList[ei,1]

		v1_idx = range(G.nodeIdx[v1,0],G.nodeIdx[v1,1])
		v2_idx = range(G.nodeIdx[v2,0],G.nodeIdx[v2,1])

		card_edge = (G.Cardinality[v1]*G.Cardinality[v2])
		assignment_map = np.zeros((card_edge,2)).astype(int)
		t = 0
		for i in xrange(G.Cardinality[v1]):
			for j in xrange(G.Cardinality[v2]):
				assignment_map[t,0] = i
				assignment_map[t,1] = j
				t +=1

		assert t==len(range(G.edgeIdx[ei,0],G.edgeIdx[ei,1])),"Mismatched length of edge potential vector"
		edge_pot_st_idx = G.edgeIdx[ei,0]
		for i in xrange(G.Cardinality[v1]):
			#get all locations in edge potentials where v1 takes value i
			val_v2 = np.where(assignment_map[:,0]==i)[0].tolist()
			idx_v2 = (edge_pot_st_idx + np.array(val_v2)).tolist()
			name = '|mu(v'+str(v1)+'='+str(i)+') = sum over { mu(v'+str(v1)+'='+str(i)+', v'+str(v2)+'='+str(val_v2)+')  } '+str("+".join([str(t) for t in idx_v2]))+'='+str(v1_idx[i])+'|'
			model.addConstr(quicksum([var[k] for k in idx_v2]),GRB.EQUAL,var[v1_idx[i]],name)

		for i in xrange(G.Cardinality[v2]):
			#get all locations in edge potentials where v2 takes value i
			val_v1 = np.where(assignment_map[:,1]==i)[0].tolist()
			idx_v1 = (edge_pot_st_idx + np.array(val_v1)).tolist()
			name = '|mu(v'+str(v2)+'='+str(i)+') = sum over { mu(v'+str(v2)+'='+str(i)+', v'+str(v1)+'='+str(val_v1)+')  } '+str("+".join([str(t) for t in idx_v1]))+'='+str(v2_idx[i])+'|'
			model.addConstr(quicksum([var[k] for k in idx_v1]),GRB.EQUAL,var[v2_idx[i]],name)

	#Set MAXTHREADS here
	model.setParam(GRB.Param.Threads, 15)
	if alg_params['maxSecondsPerMAP']>-1:
		model.setParam(GRB.Param.TimeLimit,alg_params['maxSecondsPerMAP'])
	if 'grb_log_file' in alg_params:
		model.setParam('OutputFlag',False)
		if os.path.exists(alg_params['grb_log_file']):
			os.unlink(alg_params['grb_log_file'])
		model.setParam('LogFile',alg_params['grb_log_file'])
		print "Logging to ",alg_params['grb_log_file']
		model.setParam('LogToConsole', 0)
	model.update()
	return model

#Return status of the solver
def solverStatus(model):
	solver_status = model.getAttr(GRB.Attr.Status)
	if (solver_status == GRB.LOADED):
		status = 'aborted'
		message = 'Model is loaded, but no solution information is availale.'
		term_cond = 'unsure'
	elif (solver_status == GRB.OPTIMAL):
		status = 'ok'
		message = 'Model was solved to optimality (subject to tolerances), and an optimal solution is available.'
		term_cond = 'optimal'
	elif (solver_status == GRB.INFEASIBLE):
		status = 'warning'
		message = 'Model was proven to be infeasible.'
		term_cond = 'infeasible'
	elif (solver_status == GRB.INF_OR_UNBD):
		status = 'warning'
		message = 'Problem proven to be infeasible or unbounded.'
		term_cond = 'infeasible' # Coopr doesn't have an analog to "infeasible or unbounded", which is a weird concept anyway.
	elif (solver_status == GRB.UNBOUNDED):
		status = 'warning'
		message = 'Model was proven to be unbounded.'
		term_cond = 'unbounded'
	elif (solver_status == GRB.CUTOFF):
		status = 'aborted'
		message = 'Optimal objective for model was proven to be worse than the value specified in the Cutoff  parameter. No solution information is available.'
		term_cond = 'minFunctionValue'
	elif (solver_status == GRB.ITERATION_LIMIT):
		status = 'aborted'
		message = 'Optimization terminated because the total number of simplex iterations performed exceeded the value specified in the IterationLimit parameter.'
		term_cond = 'maxIterations'
	elif (solver_status == GRB.NODE_LIMIT):
		status = 'aborted'
		message = 'Optimization terminated because the total number of branch-and-cut nodes explored exceeded the value specified in the NodeLimit parameter.'
		term_cond = 'maxEvaluations'
	elif (solver_status == GRB.TIME_LIMIT):
		status = 'aborted'
		message = 'Optimization terminated because the time expended exceeded the value specified in the TimeLimit parameter.'
		term_cond = 'maxTimeLimit'
	elif (solver_status == GRB.SOLUTION_LIMIT):
		status = 'aborted'
		message = 'Optimization terminated because the number of solutions found reached the value specified in the SolutionLimit parameter.'
		term_cond = 'stoppedByLimit'
	elif (solver_status == GRB.INTERRUPTED):
		status = 'aborted'
		message = 'Optimization was terminated by the user.'
		term_cond = 'error'
	elif (solver_status == GRB.NUMERIC):
		status = 'error'
		message = 'Optimization was terminated due to unrecoverable numerical difficulties.'
		term_cond = 'error'
	else:
		status = 'error'
		message = 'Unknown return code from GUROBI model.getAttr(GRB.Attr.Status) call'
		term_cond = 'unsure'
	return status,message,term_cond

if __name__ == '__main__':
	print "TODO: Setup test case to check this"
