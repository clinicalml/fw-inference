import os,sys
# ./exact <dir>/<name>.fg 0 > <dir>/<name>.out

#Function to compute the exact marginals
#rootDir : root directory to look for folders
#convExec: executable to convert from uai to fg
#exactExec:executable to run JT using libDAI
def computeExact(rootDir,convExec,exactExec):


	cmdNum = 1
	print "\tCompute fg representation"
	#Run commands to convert  
	cmdlist = []
	modeldir = rootDir+'/models'
	#Check if models directory exists
	if not os.path.exists(rootDir+'/models'):
		assert False,"models directory not found. Investigate"	
	for dirName, subdirList, fileList in os.walk(modeldir):
		for fname in fileList:
			basename = fname.rsplit('.',1)[0]
			extension = fname.rsplit('.',1)[1]
			if extension.strip()=="uai":
				cmd = convExec+' '+modeldir+'/'+basename+' 0 1' 
				print cmd
				os.system(cmd)
			cmdNum +=1
	#Run commands to compute exact 
	print "\tRun JT using libDAI"
	outputdir = rootDir +'/exact_out'
	if os.path.exists(outputdir)==False:
		os.mkdir(outputdir)
	for dirName, subdirList, fileList in os.walk(modeldir):
		for fname in fileList:
			#print fname,fname.rsplit('.',1)
			basename = fname.rsplit('.',1)[0]
			extension = fname.rsplit('.',1)[1]
			if extension.strip()=="fg":
				cmd = exactExec+' '+modeldir+'/'+fname+' 0 > '+outputdir+'/'+basename.split('.0')[0].strip()+'.out' 
				print cmd
				os.system(cmd)
			cmdNum +=1


if __name__ == '__main__':
	rootDir = '../Trees'
	convExec = '../libDAI/utils/uai2fg'
	exactExec = '../libDAI/examples/exact'
	computeExact(rootDir,convExec,exactExec)
