#Driver to setup experiments
import sys,os
import computeExact as step1a
import exactToMAT as step1b
import convertUAItoMAT as step2
import runPerturbAndMAP as step3

#Constants to be changed
dataDir = '../'

#Select input folder
folderName = dataDir + 'Small'
convExec = '/data/ml2/rahul/software/libDAI_trwbp_opt/utils/uai2fg'
exactExec = '../libdai-tools/exact'

#Fill out which steps are required
stepsToDo = [1,2,3]

if 1 in stepsToDo:
	print "------ Step 1 -------"
	try:
		print "Converting to FG and computing exact marginals"
		step1a.computeExact(folderName,convExec,exactExec)
		step1b.exactToMAT(folderName)
	except Exception as inst:
		print type(inst)
		print inst
	else:
		print "Done Step 1"
if 2 in stepsToDo:
	print "------ Step 2 -------"
	try:
		print "Converting UAI to MATLAB format"
		step2.convertUAItoMAT(folderName)
	except Exception as inst:
		print type(inst)
		print inst
	else:
		print "Done Step 2"

if 3 in stepsToDo:
	print "------ Step 3 -------"
	try:
		print "Running perturbMAP to compute marginals & saving to MATLAB format"
		#numSamples=5
		numSamples=10
		step3.runPerturbAndMAP(folderName,numSamples)
	except Exception as inst:
		print type(inst)
		print inst
	else:
		print "Done Step 3"
