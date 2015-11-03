#Contains the objective functions being optimized
from __future__ import division
import sys,warnings,ipdb
from math import *
import numpy as np
from util import *
from scipy.optimize import check_grad

#Vectorized Objectives
#TRW Objective
def TRW(mus,params,withGrad=False):
	#Optimized version of the TRW objective
	pot       = params['pot']
	K_c_vec   = params['K_c_vec']

	assert mus.shape==K_c_vec.shape,'mismatch shape'
	entropy =K_c_vec*computeEntropy(mus)
	f =  np.dot(mus,pot) +np.sum(entropy)
	if withGrad:
		grad = pot - (K_c_vec*(1+np.log(mus+1e-8)))
		return -1*f,-1*grad
	else:
		return -1*f
#Unvectorized Objectives
#TRW objective
def TRW_NOTOPT(mus,params,withGrad=False):
	nEdges    = params['nEdges']
	nVertices = params['nVertices']
	rhos_node = params['rhos_node']
	rhos_edge = params['rhos_edge']
	edgeIdx   = params['edgeIdx']
	nodeIdx   = params['nodeIdx']
	pot       = params['pot']
	if np.any(mus<0):
		assert False,'Negative pseudomarginals'

	if withGrad:
		grad = np.zeros(mus.shape)

	#f = -1*np.dot(mus[0:2*nVertices],pot[0:2*nVertices])
	f = -1*np.dot(mus,pot)
	#Evalate entropy over edges
	for ei in xrange(nEdges):
		K_c = rhos_edge[ei][0]
		st_idx  = edgeIdx[ei,0]
		end_idx = edgeIdx[ei,1]

		curr_mu = mus[st_idx:end_idx]
		curr_pot= pot[st_idx:end_idx]
		if withGrad:
			grad_e = -1*curr_pot + K_c*(1+np.log(curr_mu))
			grad[st_idx:end_idx] = grad_e

		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			entropy = np.multiply(curr_mu,np.log(curr_mu))*K_c
			hasNans = np.isnan(entropy)
			if np.any(hasNans):
				entropy[hasNans] = 0

		if np.any(np.isnan(entropy)):
			assert False, "NaN's found in entropy. Investigate"
		f = f + np.sum(entropy)

	#Evaluate entropy over vertices/nodes
	for vi in xrange(nVertices):
		K_c = (1-rhos_node[vi])[0]
		st_idx  = nodeIdx[vi,0]
		end_idx = nodeIdx[vi,1]

		curr_mu = mus[st_idx:end_idx]
		curr_pot= pot[st_idx:end_idx]
		if withGrad:
			grad_v = -1*curr_pot + K_c*(1+np.log(curr_mu))
			grad[st_idx:end_idx] = grad_v

		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			entropy = np.multiply(curr_mu,np.log(curr_mu))*K_c
			hasNans = np.isnan(entropy)
			if np.any(hasNans):
				entropy[hasNans] = 0

		f = f + np.sum(entropy)

		if np.any(np.isnan(entropy)):
			assert False, "NaN's found in entropy. Investigate"
	#Return according to accepted parameters
	if withGrad:
		return f,grad
	else:
		return f

#Helper Functions
#Check local consistency of mus
def checkLocalConsistency(mus,params,graphType):
	nEdges    = params['nEdges']
	nVertices = params['nVertices']
	edgeIdx   = params['edgeIdx']
	nodeIdx   = params['nodeIdx']
	pot       = params['pot']
	edgeList  = params['edgeList']

	if graphType == 'dir':
		nEdges = nEdges/2
	node_mus = mus[:nodeIdx[-1,1]].reshape(nVertices,2)
	edge_mus = mus[nodeIdx[-1,1]:].reshape(nEdges,4)

	#Sum to 1
	assert np.all(node_mus.sum(axis=1)-1<1e-12) and np.all(edge_mus.sum(axis=1)-1<1e-12),'Sum to 1 violated'

	#Consistency
	src_mus = node_mus[edgeList[:nEdges,0],:]
	dest_mus = node_mus[edgeList[:nEdges,1],:]

	consistent_src_mus = np.vstack((edge_mus[:,0]+edge_mus[:,1],edge_mus[:,2]+edge_mus[:,3])).T
	consistent_dest_mus = np.vstack((edge_mus[:,0]+edge_mus[:,2],edge_mus[:,1]+edge_mus[:,3])).T
	if np.any(np.abs(src_mus-consistent_src_mus)>1e-12) or np.any(np.abs(dest_mus-consistent_dest_mus)>1e-12):
		ipdb.set_trace()
		assert False,'Consistency violated'

#Gradient checker
def checkGrad(mus,params,f):
	val,grad = f(mus,params,True)
	e = np.zeros(mus.shape)
	numerical_grad = np.zeros(grad.shape)
	epsilon = 1e-8
	for i in range(mus.shape[0]):
		e[i]=1
		numerical_grad[i] = (f(mus+epsilon*e,params)-f(mus-epsilon*e,params))/(2*epsilon)
		e[i]=0

	def grad_fxn(mus):
		v,g = f(mus,params,True)
		return g

	def orig_fxn(mus):
		v,g = f(mus,params,True)
		return v
	if np.any(np.abs(numerical_grad-grad)>1e-6):
		ipdb.set_trace()
		assert False,'Gradient not valid'

#Compute entropy and suppress warnings and set to 0 appropriately
#Output: return -y*np.log(y) or -y*np.log(y/x)
def computeEntropy(y,x=None):
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		if x is None:
			entropy = -1*(y*np.log(y))
		else:
			entropy = -1*(y*np.log(y/x))
	hasNans = np.isnan(entropy)
	if np.any(hasNans):
		entropy[hasNans] = 0
	return entropy

#Return a list of random binary numbers
def randBinList (n):
	return [np.random.randint(2) for b in range(1,n+1)]
