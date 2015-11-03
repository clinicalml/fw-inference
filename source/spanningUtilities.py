#Utilities for Optimization over Spanning Trees
from __future__ import division
import warnings,time
import numpy as np
from scipy.sparse import *
from util import *
import objectiveFxns as obj_fxn
from scipy.sparse.csgraph import minimum_spanning_tree


#Compute entropy (or approximations) to it
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

#Undirected graphs
#Compute minimum spanning tree with -MI
def computeMST(G,mus):
	#Setup weight matrix - Inefficient
	weight_mat = lil_matrix((G.nVertices,G.nVertices))
	EdgeList = G.Edges[:,:2].astype(int)
	gradients = np.zeros((G.Edges.shape[0],))
	#Computing mutual information for all edges
	for ei in xrange(G.nEdges):
		v1 = EdgeList[ei,0]
		v2 = EdgeList[ei,1]
		var_s_i = mus[G.nodeIdx[v1,0]:G.nodeIdx[v1,1]]
		var_t_j = mus[G.nodeIdx[v2,0]:G.nodeIdx[v2,1]]
		mus_edge = mus[G.edgeIdx[ei,0]:G.edgeIdx[ei,1]]
		MI = 0
		ctr = 0
		for i in xrange(G.Cardinality[v1]):
			for j in xrange(G.Cardinality[v2]):
				MI = MI + mus_edge[ctr]*np.log(mus_edge[ctr]/(var_s_i[i]*var_t_j[j]))
				#print MI
				ctr = ctr + 1
		#Deal with numerical issues in the computation of MI
		if abs(MI)<np.exp(-10):
			MI = abs(MI)
		assert MI>=0,"Negative mutual information"+str(mus_edge)+" "+str(var_s_i)+" "+str(var_t_j)
		#Run minimum spanning tree algorithm with MI
		weight_mat[v1,v2] = -1*MI
		weight_mat[v2,v1] = -1*MI
		gradients[ei]=-1*MI
	spanning_tree = minimum_spanning_tree(weight_mat.tocsr())
	#Make this sparse
	spanning_tree_polytope_vertex = np.zeros((G.nEdges,))
	#for each of the edges, check if its in the minimum spanning tree
	for ei in xrange(G.nEdges):
		v1 = EdgeList[ei,0]
		v2 = EdgeList[ei,1]
		if spanning_tree[v1,v2]!=0 or spanning_tree[v2,v1]!=0:
			spanning_tree_polytope_vertex[ei]=1
	return spanning_tree_polytope_vertex,gradients
