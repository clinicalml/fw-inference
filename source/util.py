#Utilities used by all files
from __future__ import division
import os,ipdb,copy
import numpy as np
from scipy.io import *
from scipy.sparse import *
from scipy.linalg import *
import cPickle as pickle

#Create directories if absent
#Input: List of directories
#Output: None
def createIfAbsent(dirList):
	for d in dirList:
		if not os.path.exists(d):
			os.mkdir(d)

def createMarginalVector(solution,List):
	nodes = np.tile(solution,(2,1)).T
	nodes[:,0] = 1-nodes[:,0]
	nodes = nodes.ravel()
	nEdges = List.shape[0]
	src = solution[List[:nEdges,0].astype(int)]
	dest= solution[List[:nEdges,1].astype(int)]
	result = np.vstack((src,dest)).T
	edge_idx = result[:,0]*2+result[:,1]
	edges = np.zeros(List[:nEdges,2:].shape)
	#Set relevant positions to 1
	#0
	edges[edge_idx==0,0]=1
	#1
	edges[edge_idx==1,1]=1
	#2
	edges[edge_idx==2,2]=1
	#3
	edges[edge_idx==3,3]=1
	return np.hstack((nodes,edges.ravel()))


def createPotentialVector(List,Fields):
	return np.hstack((Fields.ravel(),List[:,2:].ravel()))

#Write a marginal vector to UAI format
#Assumes potentials in marginal_vec are strictly positive (i.e *NOT* in log format)
#Input : outputfilename (File name to write UAI file to)
#		 marginal_vec (marginal vector to write UAI file with )
#		 N (number of variables)
#		 List (edges in graph and their corresponding edge potentials)
#Output: None, writes to file
def writeToUAI(outputfilename,marginal_vec,N,Cardinality,EdgeList):
	f = open(outputfilename,'w',0)
	f.write('MARKOV\n')
	f.write(str(N)+'\n')
	vert_card = np.squeeze(Cardinality).astype(int).tolist()
	f.write(" ".join([str(k) for k in vert_card]))
	graph_str = ""

	num_lines = 0

	#Write Stage 1 of UAI File
	for i in xrange(N):
		graph_str+= "1 "+str(i)+"\n"
		num_lines+=1

	for ei in xrange(EdgeList.shape[0]):
		graph_str+= "2 "+str(EdgeList[ei,0])+" "+str(EdgeList[ei,1])+"\n"
		num_lines+=1

	f.write("\n"+str(num_lines)+"\n")
	f.write(graph_str+"\n")

	#Write Stage 2 of UAI File
	obj_str = ""
	idx = 0
	for i in xrange(N):
		m_vec = [str(m) for m in marginal_vec[idx:idx+vert_card[i]].tolist()]
		obj_str+= str(vert_card[i])+"\n"+" ".join(m_vec)+"\n\n"
		idx+=vert_card[i]
	for ei in xrange(EdgeList.shape[0]):
		edge_card = vert_card[EdgeList[ei,0]]*vert_card[EdgeList[ei,1]]
		m_vec = [str(m) for m in marginal_vec[idx:idx+edge_card].tolist()]
		obj_str+= str(edge_card)+"\n"+" ".join(m_vec)+"\n\n"
		idx+=edge_card

	f.write(obj_str)
	f.close()
	assert idx==len(marginal_vec)

#load UAI data from matlab file
#Input : filename
#Output: Data loaded from matfile
def loadUAIfromMAT(inputfilename):
	inputfile = open(inputfilename,'rb')
	data = loadmat(inputfile, mdict=None)
	inputfile.close()
	card = np.squeeze(data['Cardinality']).astype(int)
	# N,Nodes,Edges,Cardinality,mode = loadUAIfromMAT(inputfilename)
	return np.squeeze(data['N'][0][0]),np.squeeze(data['Fields']),np.squeeze(data['List']),card,'UAI'

#Write out a graph structure to file
#You can re-use this to generate the spanning trees of the graph afterwards
#Input : N (number of vertices), EdgeList (List of edges), Weights (Weights associated with each edge)
#Output: File written out
#Format of file
#Number of vertices,Edges and Associated Weights
def writeGraph(fname,N,EdgeList,Weights,NodeWeights=None):
	fout = open(fname,'w')
	fout.write(str(N)+'\n')
	if NodeWeights is not None:
		fout.write("\n".join([str(t) for t in NodeWeights.tolist()])+"\n")
	assert Weights.shape[0]==EdgeList.shape[0],'Weights and Edges do not have the same size'
	for ei in xrange(EdgeList.shape[0]):
		fout.write(str(EdgeList[ei,0])+','+str(EdgeList[ei,1])+','+str(Weights[ei])+'\n')
	fout.close()

#Graph class : container for all graph based structures and devices
#IMPORTANT : All potentials stored in G are in *LOG* format, this is assumed to be passed in
class Graph:
	def __init__(self, *args, **kwargs):
		#Parse arguments
		#Step 1: Check how the constructor is called
		if len(kwargs.keys())==0:
			assert False,"Specify all arguments for constructor as arg1=val1, arg2=val2"
		else:
			#Check arguments for constructor
			assert 'mode' in kwargs and 'weighted' in kwargs and 'type' in kwargs, 'Specify mode(<UAI>,<minimal>), weighted(<True>,<False>), type(<dir>,<undir>) in arguments'
			mode = kwargs['mode']
			weighted = kwargs['weighted']
			graphType= kwargs['type']
			self.graphFile = './graphFile.grph'
			#Create a graphFile variable if specified
			if 'graphFile' in kwargs:
				self.graphFile = kwargs['graphFile']
			assert mode == 'UAI',("Unrecognized mode: "+mode)
			assert type(weighted) is bool,'Weighted argument is required to be boolean'
			assert graphType == 'undir','GraphType not specified correctly'
		#Step 2: Setup for UAI mode or for minimal mode
		#Attributes:
		#N (Number of vertices), Edges (E x (2+CardEdge))
		#Nodes (Node potentials) Cardinality (Cardinality of variables)
		#Weighted (Whether or not to use the weighted graph for edge probabilities)
		#graphType ('dir' or 'undir')
		if mode == 'UAI':
			assert 'N' in kwargs and 'Edges' in kwargs \
				and 'Nodes' in kwargs and 'Cardinality' in kwargs,"Insufficient inputs for UAI mode"
			self.Edges=np.copy(kwargs['Edges'])
			#Workaround for copy constructor in the case of directed graphs
			self.Nodes=np.copy(kwargs['Nodes'])
			self.N   = self.Nodes.shape[0]
			self.Cardinality=np.reshape(np.copy(kwargs['Cardinality']),(self.N,)).astype(int)
			self.weighted = weighted
			self.graphType = graphType
			#Adjust for MATLAB indices
			if self.Edges[:,:2].min()==1:
				self.Edges[:,:2]=self.Edges[:,:2]-1
			assert np.all(self.Edges[:,:2].astype('int')>=0),'Negative edges detected. Investigate'
		else:
			assert False,'Mode not UAI. Should not be here'
		#Step 4: Create structures necessary for graph
		#1. Adjacency
		#2. Weight Matrix
		#3. rhos, rhos_node
		#4. nVertices, nEdges
		#5. var/edge begin/end
		#6. centre of marginal polytope for current graph
		self.createGraphStruct()

	#append additional features to graph
	def createGraphStruct(self):
		self.nVertices = self.N
		self.nEdges    = self.Edges.shape[0]

		#Computing the potential/initial marginal vector
		pot = []
		init_vec = []
		self.nodeIdx = np.zeros((self.nVertices,2)).astype(int)
		self.edgeIdx = np.zeros((self.nEdges,2)).astype(int)
		idx = 0
		total_card = 0
		for vi in xrange(self.nVertices):
			#Extract only upto the cardinality of the vertex
			v_card = self.Cardinality[vi]
			pot += self.Nodes[vi,:v_card].tolist()
			self.nodeIdx[vi,0]=int(idx)
			self.nodeIdx[vi,1]=int(idx+v_card)
			init_vec += v_card*[float(1)/v_card]
			assert np.array_equal(np.array(pot[self.nodeIdx[vi,0]:self.nodeIdx[vi,1]]),self.Nodes[vi,:v_card]),"Node potential vector not set"
			idx += v_card
			total_card += v_card

		for ei in xrange(self.nEdges):
			v1 = self.Edges[ei,0].astype('int')
			v2 = self.Edges[ei,1].astype('int')
			card_edge = (self.Cardinality[v1]*self.Cardinality[v2])
			#assert type(v1) is np.int64 and type(v2) is np.int64 and type(card_edge) is np.int64,"Indices not integer. Investigate"
			pot += self.Edges[ei,2:2+card_edge].tolist()
			self.edgeIdx[ei,0]=idx
			self.edgeIdx[ei,1]=idx+card_edge
			init_vec += card_edge*[float(1)/card_edge]
			assert np.array_equal(np.array(pot[self.edgeIdx[ei,0]:self.edgeIdx[ei,1]]),self.Edges[ei,2:2+card_edge]),"Edge potential vector not set correctly"
			idx += card_edge
			total_card += card_edge
		#assert that size of the marginal vector == as expected
		assert len(pot)==total_card, "Length of marginal vector does not match potential vector"
		self.pot = np.array(pot)
		self.init_vec = np.array(init_vec)

		#Use sparse matrices here.
		#TODO: These are never used later so consider freeing them
		self.Adjacency = np.zeros((self.N,self.N))
		self.Weighted  = np.zeros((self.N,self.N))
		#self.Adjacency = lil_matrix((self.N,self.N))
		#self.Weighted = lil_matrix((self.N,self.N))
		EdgeList = self.Edges[:,:2].astype(int)
		for i in xrange(self.nEdges):
			v1 = EdgeList[i,0]
			v2 = EdgeList[i,1]
			card_edge = (self.Cardinality[v1]*self.Cardinality[v2])
			assert type(v1) is np.int64 and type(v2) is np.int64 and type(card_edge) is np.int64,"Indices not integer. Investigate"
			if int(self.Cardinality[v1])==2 and int(self.Cardinality[v2])==2:
				self.Weighted[v1,v2] = np.abs(self.Edges[i,2]+self.Edges[i,5]-self.Edges[i,3]-self.Edges[i,4])
			else:
				self.Weighted[v1,v2] = np.abs(np.sum(self.Edges[i,2:2+card_edge]))
			self.Adjacency[v1,v2] = 1
		#Make matrices symmetric
		#self.Weighted = self.Weighted.tocsr()
		#self.Adjacency = self.Adjacency.tocsr()
		self.Weighted = self.Weighted + self.Weighted.transpose()
		self.Adjacency = self.Adjacency + self.Adjacency.transpose()
		#Use the above to compute the edge appearance probabilities using the Matrix Tree Theorem
		#for undirected graphs
		self.computeEdgeRhos()
		#Define containers here so they are not redefined
		self.K_c_vec = np.ones(self.pot.shape)
		self.rhos_node = np.zeros((self.nVertices,))
		self.computeNodeRhos()

	#Compute edge appearance probabilities using matrix tree theorem for undirected graphs
	def computeEdgeRhos(self):
		rhos_edge = np.zeros((self.nEdges,1))
		#check if weighted
		if self.weighted:
			mat = self.Weighted
		else:
			mat = self.Adjacency

		# #For use with numpy
		L = np.diag(np.asarray(mat).sum(axis=1))-np.asarray(mat)
		L1 = L-1
		#For use with scipy sparse matrices
		#L_sp = csr_matrix(np.diag(np.asarray(mat.sum(axis=0))[0]))-mat
		#Invert Matrix
		try:
			#Numpy
			Linv = np.linalg.inv(L1)
			#scipy sparse -> inverse results in numpy array (cant get around this)
			#Linv_sp  = inv(L_sp.todense()-1)
		except LinAlgError:
			assert False, "Matrix inverse not defined for graph laplacian"

		#mat = mat.tolil() #Faster random access
		for ei in xrange(self.nEdges):
			v1 = self.Edges[ei,0].astype('int')
			v2 = self.Edges[ei,1].astype('int')
			#res_sp = mat[v1,v2]*(Linv_sp[v1,v1]+Linv_sp[v2,v2]-2*Linv_sp[v1,v2])
			res_np = mat[v1,v2]*(Linv[v1,v1]+Linv[v2,v2]-2*Linv[v1,v2])
			#assert np.abs(res_sp-res_np)<1e-10,'err'
			rhos_edge[ei] = res_np
		self.rhos_edge= rhos_edge

	#update node probabilities
	def computeNodeRhos(self):
		self.rhos_node[:] = 0
		self.K_c_vec[:] = 1
		for ei in xrange(self.nEdges):
			v1 = self.Edges[ei,0].astype('int')
			v2 = self.Edges[ei,1].astype('int')
			self.rhos_node[v1] += self.rhos_edge[ei][0]
			self.rhos_node[v2] += self.rhos_edge[ei][0]
			self.K_c_vec[self.edgeIdx[ei,0]:self.edgeIdx[ei,1]] = self.rhos_edge[ei][0]
			self.K_c_vec[self.nodeIdx[v1,0]:self.nodeIdx[v1,1]] -= self.rhos_edge[ei][0]
			self.K_c_vec[self.nodeIdx[v2,0]:self.nodeIdx[v2,1]] -= self.rhos_edge[ei][0]
