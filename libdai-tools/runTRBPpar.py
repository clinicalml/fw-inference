#Run TRBP and save results in comparison folder
import numpy as np
import os,sys,glob,time
from scipy.io import *
from multiprocessing import Pool

def runPAR(args):
	name,cmd = args
	pid = os.getpid()
	print name," is running on pid=",pid
	start = time.time()
	os.system(cmd)
	end = time.time()
	print name," is finished on pid=",pid
	return name,end-start

def saveTRWBPresults(mainfolder,subfolder,prefix):
	TRBPfolder = mainfolder+'/'+subfolder
	assert os.path.exists(TRBPfolder),'Folder not found'
	for f in glob.glob(TRBPfolder+'/*.trbp'):
		result = np.loadtxt(f)
		matfname = f.replace('.trbp','.mat')
		mat = {}
		if os.path.exists(matfname):
			mat = loadmat(matfname)
		mat[prefix+'_logz']=result[0]
		mat[prefix+'_runtime'] =result[1]
		mat[prefix+'_marg'] = result[2:].ravel()
		print "Saving to",matfname
		savemat(matfname,mat)

def runTRBP_opt(mainfolder):
	#With tightening
	TRBP_bin = './trwbp'
	TRBPfolder = mainfolder+'/TRWBP_opt'
	if not os.path.exists(TRBPfolder):
		os.mkdir(TRBPfolder)
	assert os.path.exists(mainfolder+'/models')
	assert os.path.exists(TRBP_bin),'trbp test binary not found'
	commands = []
	for f in glob.glob(mainfolder+'/models/*.uai'):
		basename = os.path.basename(f).replace('.uai','')
		outputname = TRBPfolder+'/'+basename
		logfilename = TRBPfolder+'/'+basename+'.libdaiout_1000'
		print "Input: ",f," Output: ",outputname
		cmd = 'echo "Starting '+f+'" & '+TRBP_bin+' '+f+' '+outputname+' 10 1000 0.9 1000 &> '+logfilename
		print "Running: ",cmd
		commands.append((os.path.basename(f), cmd))
	pool = Pool(processes = 20)
	results = pool.map(runPAR,commands,chunksize=5)
	pool.close()
	pool.join()
	for (fname,time) in results:
			print fname,' took ',time/60.,' minutes'
	saveTRWBPresults(mainfolder,'TRWBP_opt','trwbp_opt')
def runTRBP(mainfolder):
	#Without tightening
	TRBP_bin = './trwbp'
	TRBPfolder = mainfolder+'/TRWBP'
	if not os.path.exists(TRBPfolder):
		os.mkdir(TRBPfolder)
	assert os.path.exists(mainfolder+'/models')
	assert os.path.exists(TRBP_bin),'trbp test binary not found'
	commands = []
	for f in glob.glob(mainfolder+'/models/*.uai'):
		basename = os.path.basename(f).replace('.uai','')
		outputname = TRBPfolder+'/'+basename
		logfilename = TRBPfolder+'/'+basename+'.libdaiout_1000'
		print "Input: ",f," Output: ",outputname
		cmd = 'echo "Starting '+f+'" & '+TRBP_bin+' '+f+' '+outputname+' 1 1000 0.9 1000 &> '+logfilename
		print "Appending: ",cmd
		commands.append((os.path.basename(cmd),cmd))
	pool = Pool(processes = 20)
	results = pool.map(runPAR,commands,chunksize=5)
	pool.close()
	pool.join()
	for (fname,time) in results:
			print fname,' took ',time/60.,' minutes'
	saveTRWBPresults(mainfolder,'TRWBP','trwbp')
if __name__=='__main__':
	mainfolder = '../ChineseCharTruncated'
	runTRBP(mainfolder)
	runTRBP_opt(mainfolder)
