#File implements computing marginals and the partition function using the paper perturb and MAP
from __future__ import division
import sys,ipdb,glob,os
sys.path.append('../source')
import util,copy
import MAPsolver
import numpy as np
from sklearn.preprocessing import normalize
import timeit
#Return a matrix of shape with values taken from
#independant samples of a gumbel distribution
def gumbelToss(shape):
	EulerConst=0.57721566490153
	return -np.log(-np.log(np.random.random(shape)))-EulerConst


#Approximate the log partition function and extimate the marginals
def doInferencePerturbMAP(G,numSamples = 10):
	print "Running perturbMAP with ",numSamples," samples"
	#Use sampling from the gumbel distribution to esimate the log partition function
	start = timeit.default_timer()
	#logZ = estimateLogZ(numSamples,G,quiet=True)
	logZ = ubLogZ(numSamples,G,quiet=False)

	end = timeit.default_timer()
	time_logz = start-end

	mus = np.zeros((G.nVertices,2))
	DEBUG = False
	start = timeit.default_timer()
	#To estimate the marginals, fix the value of a variable,
	#estimate the log partition function using the same procedure and normalize
	all_idx = range(G.nVertices)
	for vi in range(G.nVertices):
		print "\nConditioning on vertex :",vi,
		remaining_idx = all_idx[:vi] + all_idx[vi+1:]
		vmap = {}
		for ctr,node in enumerate(remaining_idx):
			vmap[node]=ctr
		beliefs = []
		lgbeliefs = []
		for val in range(2):
			print "(",val,"),",
			#Consider fixing vi = val
			G_truncated = copy.deepcopy(G)
			#Go through the edges
			valid_edge_idx = []
			for ei in range(G.nEdges):
				#Get source and destination variables
				if G.Edges[ei,0]==vi:
					src = vi
					dest = G.Edges[ei,1].astype('int')
					if DEBUG:
						print "\nvi is src ",
					not_vi = dest
				elif G.Edges[ei,1]==vi:
					src = G.Edges[ei,0].astype('int')
					dest = vi
					if DEBUG:
						print "\nvi is dest ",
					not_vi = src
				else:
					#This edge still exists in truncated graph
					valid_edge_idx += [ei]
					continue
				assert not_vi in remaining_idx,"Other node not found in remaining vertices"
				#This edge will no longer exist in graph so add potentials to the nodes that will remain in the graph
				#appropriately
				edge_pot_to_add_idx = []
				ctr = 0
				for v1 in range(G_truncated.Cardinality[src]):
					for v2 in range(G_truncated.Cardinality[dest]):
						if (src==vi and v1 == val) or (dest==vi and v2 == val):
							edge_pot_to_add_idx += [ctr]
						ctr +=1
				if DEBUG:
					print "Appending: ",2+np.array(edge_pot_to_add_idx),
				#Subsume the effect of the edge potential into the node potential
				#Append to the node potentials, the value of the edge potentials corresponding to the values where vi takes "val"
				G_truncated.Nodes[not_vi,:] =  G_truncated.Nodes[not_vi,:] + G_truncated.Edges[ei,2+np.array(edge_pot_to_add_idx)]
			#Select only the relevant variables
			G_truncated.nVertices -= 1
			G_truncated.N -= 1
			G_truncated.Nodes = G_truncated.Nodes[remaining_idx,:]
			assert G_truncated.Nodes.shape[0]==G.nVertices-1,"One vertex conditioned on"
			G_truncated.Cardinality = G_truncated.Cardinality[remaining_idx]
			assert G_truncated.Cardinality.shape[0]==G.nVertices-1,"One vertex conditioned on"
			G_truncated.Edges = G_truncated.Edges[np.array(valid_edge_idx),:]
			#Need to modify EdgeList since a vertex was removed
			for ei in range(G_truncated.Edges.shape[0]):
				v1 = G_truncated.Edges[ei,0].astype('int')
				v2 = G_truncated.Edges[ei,1].astype('int')
				assert v1!=vi and v2!=vi,"Removed vertex found in edge list. Investigate"
				G_truncated.Edges[ei,0] = vmap[v1]
				G_truncated.Edges[ei,1] = vmap[v2]
			G_truncated.createGraphStruct()
			G_truncated.nEdges = G_truncated.Edges.shape[0]
			#logZest = estimateLogZ(numSamples,G_truncated,quiet=True)
			logZest =ubLogZ(numSamples,G_truncated,quiet=True)
			#Add the contribution of the single node potential to the log partition function estimated
			lgbeliefs.append(G.Nodes[vi,val]+logZest)
			beliefs.append(np.exp(G.Nodes[vi,val]+logZest))

		#Normalize and append
		#tmp = np.array(beliefs)
		#tmp/tmp.sum()
		M = np.max(lgbeliefs)
		mus[vi,:] =np.exp(lgbeliefs-(M+np.log(np.exp(lgbeliefs-M).sum())))
	print "\nEstimated log partition : ",logZ," \nEstimated marginals\n",np.array(mus)
	end = timeit.default_timer()
	time_marg = start - end
	return logZ,mus.ravel(),time_marg,time_logz

#Normalize beliefs using the log sum exp trick
#URL (for future reference) : https://hips.seas.harvard.edu/blog/2013/01/09/computing-log-sum-exp/
def lgsumexp(mus):
	M = np.max(mus)
	return M+np.log(np.exp(mus-M).sum())

def ubLogZ(numSamples,G,quiet = False):
	energy = 0
	for s in xrange(numSamples):
		G_cp = copy.deepcopy(G)	
		G_cp.Nodes+=gumbelToss(G_cp.Nodes.shape)
		theta_expanded = np.hstack((G_cp.Nodes.ravel(),G_cp.Edges[:,2:].ravel()))
		#Run MAP inference
		marginal_polytope_vertex,toulbar2Output,MAPsoln = MAPsolver.runMAP(np.array(theta_expanded),
				G_cp.N,G_cp.Edges,G_cp.Cardinality,'undir',uai_fname = './temp.uai',timer=-1)
		energy += np.dot(marginal_polytope_vertex,theta_expanded)
	energy /= float(numSamples)
	return energy

#Use toulbar2 to perform MAP inference and return result
def estimateLogZ(numSamples,G,quiet=False):
	#Mapping from variables old graph to new one
	inflation_mapping = {}
	inflated_nVertices = 0
	for i in range(G.nVertices):
		inflation_mapping[i] = []
		for j in range(numSamples):
			inflation_mapping[i]+= [inflated_nVertices]
			inflated_nVertices+=1
	assert inflated_nVertices == G.nVertices*numSamples,"Number of inflated variables does not match the number of samples"
	#Setup cardinality
	inflated_Cardinality = 2*np.ones((numSamples*G.nVertices,)).astype('int')
	#Setup marginal vector

	#Node potentials
	avg_node = (float(1)/numSamples)
	toss = gumbelToss((2*G.nVertices*numSamples,))
	theta_expanded = (np.tile(G.Nodes,(1,numSamples)).ravel()+toss)*avg_node

	theta_expanded = theta_expanded.tolist()
	#Setup edge list
	inflated_List = np.zeros((G.nEdges*numSamples*numSamples,6))
	inflated_edge_num = 0
	EdgeList = G.Edges[:,:2].astype(int)
	#Average the potentials across the edges
	avg_edge = (float(1)/(numSamples*numSamples))
	for ei in xrange(G.nEdges):
		v1 = EdgeList[ei,0]
		v2 = EdgeList[ei,1]
		for src in inflation_mapping[v1]:
			for dest in inflation_mapping[v2]:
				#If v1-v2, then all pairs of
				#mapping[v1]-mapping[v2] will be connected with the same edge potentials
				inflated_List[inflated_edge_num,0] = src
				inflated_List[inflated_edge_num,1] = dest
				inflated_List[inflated_edge_num,2:] = G.Edges[ei,2:]
				#Consider also tossing the edge potentials
				#theta_expanded += ((G.Edges[ei,2:]+gumbelToss((4,)))*avg_edge).tolist()
				theta_expanded += ((G.Edges[ei,2:])*avg_edge).tolist()
				inflated_edge_num+=1

	assert inflated_edge_num == G.nEdges*numSamples*numSamples,"The number of inflated edges is off"+ \
			str(inflated_edge_num)+' vs '+str(G.nEdges*numSamples*numSamples)

	if not quiet:
		print "Step 1: Creating inflated graph."

	#Step 2] Run MAP on the inflated graphical model (in this case using toulbar2)
	if not quiet:
		print "Step 2: MAP inference with toulbar2"
	marginal_polytope_vertex,toulbar2Output,MAPsoln = MAPsolver.runMAP(np.array(theta_expanded),
			inflated_nVertices,inflated_List,inflated_Cardinality,
			'undir',uai_fname = './temp.uai',timer=-1)
	#Delete all temporary files
	print "Deleting: ",
	for f in glob.glob("temp*"):
		print f,
		os.unlink(f)
	#Step 3] Compute the energy of the MAP solution as the estimate of logZ
	if not quiet:
		print "Step 3: Energy of MAP solution is estimate of logZ"
	energyEstimate = np.dot(marginal_polytope_vertex,theta_expanded)
	if not quiet:
		print "Computing energy estimate"
	return energyEstimate

if __name__ == '__main__':
	print "10 Node Complete"
	inputfilename = '/data/ml2/rahul/FrankWolfeMarginalInf/Synthetic/uai_mat/810.mat'
	N,Nodes,Edges,Cardinality,mode = util.loadUAIfromMAT(inputfilename)
	G = util.Graph(N=N,Edges=Edges,Nodes=Nodes,Cardinality=Cardinality,mode=mode,weighted=True,type='undir')
	print doInferencePerturbMAP(G,numSamples = 10)
	print ubLogZ(20,G,quiet = True)
#	print "Debug 3 node"
#	inputfilename = '/data/ml2/rahul/FrankWolfeMarginalInf/Debug/uai_mat/grid3x3.mat'
#	N,Nodes,Edges,Cardinality,mode = util.loadUAIfromMAT(inputfilename)
#	G = util.Graph(N=N,Edges=Edges,Nodes=Nodes,Cardinality=Cardinality,mode=mode,weighted=True,type='undir')
#	print doInferencePerturbMAP(G,10)
