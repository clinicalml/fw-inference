from scipy.io import *
import numpy as np
import networkx as nx
import os
import sys

display=False
if display:
	import matplotlib.pyplot as plt

if __name__ == '__main__':
	rootDir = '../Trees'
	convertUAItoMAT(rootDir)

def convertUAItoMAT(rootDir):
	modeldir = rootDir+'/models'
	outdir = rootDir+'/uai_mat'
	if not os.path.exists(outdir):
			os.mkdir(outdir)
	for dirName, subdirList, fileList in os.walk(modeldir):
		for fname in fileList:
			basename = fname.rsplit('.',1)[0]
			extension = fname.rsplit('.',1)[1]
			if extension.strip()=="uai":
				UAIfile = modeldir+'/'+basename+'.uai'
				MARfile = modeldir+'/'+basename+'.uai.MAR'
				MATfile = outdir+'/'+basename+'.mat'
				print "Running convert for: ",UAIfile,MARfile,MATfile
				#convert to matlab format
				convertToMat(UAIfile,MARfile,MATfile)

def getNums(data,sep):
	l = data.strip().split(sep)
	l_fl = [float(t) for t in l]
	return np.array(l_fl)

#initialize theta and draw graph
def getTheta(f,variable_card):
	theta = []
	#start from line 4
	G = nx.Graph()

	print "Processing for theta:"
	cardinality_vars = [int(v) for v in variable_card]
	init_num_vars = len(cardinality_vars)
	max_var_card = max(cardinality_vars)
	#print "Cardinality Vars: ",cardinality_vars
	num_lines = int(f.readline())

	var_list = []
	edge_list = []

	#Cardinality for vars/edges for which potentials are defined
	cardinality_pot = []
	pot_idx = 0
	type_map = {}
	#Parse through num_lines to get create the graph
	print "\tStage 1: ",num_lines," lines"
	for i in range(num_lines):
		l = f.readline()
		l =l.replace(" ","\t")
		elems = l.strip().split('\t')
		clique_size = int(elems[0])
		if clique_size==1:
			var_num = int(elems[1])
			cardinality_pot.append(cardinality_vars[var_num])
			G.add_node(var_num)
			var_list.append(var_num+1)
			type_map[pot_idx] = 'v'
		elif clique_size==2:
			#edge from v1 to v2
			v1 = int(elems[1])
			v2 = int(elems[2])

			cardinality_pot.append(cardinality_vars[v1]*cardinality_vars[v2])
			G.add_edge(v1,v2)
			edge_list.append((v1+1,v2+1))
			type_map[pot_idx] = 'e'
		else:
			sys.exit("A] Non-Pairwise MRF not implemented. Exiting....")
		pot_idx +=1

	num_vars = len(var_list)
	num_edges = len(edge_list)

	print "Num Vars with Potentials: ",num_vars," Num Edges: ",num_edges
	print "Maximum Var Cardinality ",max_var_card

	#Set variables to have zero potentials as default
	Fields = np.zeros((max(num_vars,init_num_vars),max_var_card))

	Edges = np.zeros((num_edges,2+(max_var_card*max_var_card)))

	print Fields.shape
	print Edges.shape
	#print "Cardinality for which Potentials are defined: ",cardinality_pot
	print "\tStage 2:Creating theta"
	clique = 0
	var_idx = 0
	edge_idx = 0
	done =0

	#mode = 0 expects cardinality, mode = 1 expects to read potentials
	pot_idx = 0
	mode = 0
	clique_card = -1
	line_num=1
	#Assumes UAI file does not contain cliques
	while done==0:
		data = f.readline().strip()
		line_num+=1
		if line_num%5000==0:
			print "Done:",line_num
		if data=="":
			continue
		data = data.replace(" ","\t")
		if mode==0:
			clique_card = int(data)
			#Check if the cardinality found is the same as from the previous section
			assert clique_card == cardinality_pot[pot_idx],str(clique_card)+ " vs "+str(cardinality_pot[pot_idx])+" : Cardinality does not match up"

			#Set the number of potentials
			pot_ctr = 0
			mode = 1
		else:
			#Get the data
			#data_arr = np.fromstring(data,sep='\t',dtype=np.float64)
			data_arr = getNums(data,'\t')
			l = max(data_arr.shape)
			#print ""
			#print "Potential Index: ",pot_idx,"Total # Pot: ",len(cardinality_pot)," Type: ",type_map[pot_idx]
			#print "Clique Card: ",clique_card," Pot_Ctr (filled so far):",pot_ctr
			#print "# data points: ",l
			#print "Data: ",data_arr
			#print "-----------"

			if type_map[pot_idx]=='v':
				Fields[var_idx,pot_ctr:pot_ctr+l] = data_arr
			elif type_map[pot_idx]=='e':
				Edges[edge_idx,0] = edge_list[edge_idx][0]
				Edges[edge_idx,1] = edge_list[edge_idx][1]
				Edges[edge_idx,2+pot_ctr:2+pot_ctr+l] = data_arr
			else:
				sys.exit("B] Non-Pairwise MRF not implemented. Exiting....")
			pot_ctr = pot_ctr + l
			#Check if all the numbers have been processed
			if pot_ctr == clique_card:
				#Reset mode, move onto next edge
				mode = 0
				if type_map[pot_idx]=='v':
					var_idx +=1
					Fields
				elif type_map[pot_idx]=='e':
					edge_idx += 1
				else:
					sys.exit("C] Non-Pairwise MRF not implemented. Exiting....")
				pot_idx += 1
		#Done if every potential has been assigned
		if pot_idx == len(cardinality_pot):
			done = 1
	assert var_idx==num_vars and edge_idx==num_edges, "All potentials not considered. Investigate"

	remaining_lines = [t.strip() for t in f.readlines()]
	for l in remaining_lines:
		if l!="":
			assert False,"Data to be parsed :"+l

	if display:
		nx.draw(G)
		#plt.savefig(DATASETNAME+".png") # save as png
		plt.show() # display

	return theta,Fields,Edges

def getMarginalSolution(l):
	print "Processing Solution"
	solution_mus = []
	idx = 0
	num_var = int(l[idx])
	idx +=1
	print "\t",num_var," variables"
	var = 1
	while var <= num_var:
		print "\tVariable ",var
		cardinality = int(l[idx])
		idx+=1
		for t in range(cardinality):
			solution_mus.append(float(l[idx]))
			idx +=1
		var += 1
	return num_var,solution_mus

def getSolution(MARfile):
	#parse solution to extract solution_mus
	if not os.path.exists(MARfile):
		return -1,[-1]
	f = open(MARfile,'r')
	l1 = f.readline()
	assert "MAR" in l1,"Marginal solution not found. Exiting"
	l2 = f.readline()
	assert len(l2.strip().split(' '))<2, "Second line contains information. Investigate"
	l3 = f.readline()
	f.close()

	N_solution,solution_mus = getMarginalSolution(l3.strip().split(' '))
	return N_solution,solution_mus

def convertToMat(UAIfile,MARfile,MATfile):
	N_solution,solution_mus = getSolution(MARfile)
	print solution_mus

	f = open(UAIfile,'r')
	l1 = f.readline()
	assert "MARKOV" in l1,"Non Markov Network. Exiting"
	l2 = f.readline()
	N_uai =int(l2.strip())
	if N_solution > 0:
		assert N_solution==N_uai, "Variables do not match from solution.Investigate"
	l3 = f.readline()
	assert N_uai==len(l3.strip().split(' ')), "Incompatible number of variables"
	cardinality = [int(t) for t in l3.strip().split(' ')]
	#theta is unused.
	theta,Fields,List = getTheta(f,l3.strip().split(' '))
	f.close()

	#MATLAB code assumes log potentials but UAI in potential format
	#Conver to log potentials
	Fields = np.log(Fields)
	List[:,2:]=np.log(List[:,2:])
	#Set -inf to large negative value
	Fields[np.isneginf(Fields)]=-20000
	List[np.isneginf(List)]=-20000

	print "Saving MATLAB file"
	mat = {}

	mat['N'] = N_uai
	mat['theta'] = theta
	mat['Fields'] = Fields
	mat['List'] = List
	mat['Cardinality'] = cardinality
	#print Fields
	#print List
	#print theta
	#print cardinality
	mat['solution_mus'] = solution_mus
	savemat(MATfile, mat)
	fname = MATfile.split('.')[0].strip()
	#np.savetxt(fname+"-Cardinality.csv", cardinality, delimiter=",",fmt='%g')
	#np.savetxt(fname+"-List.csv", List, delimiter=",",fmt='%g')
	#np.savetxt(fname+"-Fields.csv", Fields, delimiter=",",fmt='%g')
	print "Done Conversion"



