//Load opengm mode specified, run MAP solver
//and write results to output file
#include "map_tools.h"

//Interface to OpenGM's MAP solvers


//LazyFlipper
#include <opengm/inference/lazyflipper.hxx>
typedef opengm::LazyFlipper<GraphicalModelType, opengm::Minimizer> LazyFlipper;

//QPBO
#include <opengm/inference/external/qpbo.hxx> 
typedef opengm::external::QPBO<GraphicalModelType> QPBO;

//TRWS
#include <opengm/inference/external/trws.hxx> 
typedef opengm::external::TRWS<GraphicalModelType> TRWS;

//DualDecomposition
#include <opengm/inference/messagepassing/messagepassing.hxx> 
#include <opengm/inference/dualdecomposition/dualdecomposition_subgradient.hxx> 
#include <opengm/inference/dualdecomposition/dualdecomposition_bundle.hxx>
typedef double ValueType;
typedef opengm::Minimizer AccType;
typedef opengm::DDDualVariableBlock<marray::Marray<ValueType> > DualBlockType; 
typedef opengm::DualDecompositionBase<GraphicalModelType,DualBlockType>::SubGmType SubGmType;
typedef opengm::BeliefPropagationUpdateRules<SubGmType, AccType> UpdateRuleType;
typedef opengm::MessagePassing<SubGmType, AccType, UpdateRuleType, opengm::MaxDistance> InfType; 
typedef opengm::DualDecompositionSubGradient<GraphicalModelType,InfType,DualBlockType> DualDecompositionSubGradient;

//LSATR
#include <opengm/inference/lsatr.hxx>
typedef opengm::LSA_TR<GraphicalModelType, opengm::Minimizer> LSATR;

//ICM
#include <opengm/inference/icm.hxx>
typedef opengm::ICM<GraphicalModelType, opengm::Minimizer> ICM; 

//Write MAP result to file
void writeResult(std::vector<LabelType> result,std::string ofname)
{
	std::ofstream output;
	output.open(ofname.c_str());
	for (size_t i = 0;i<result.size();i++)
	{
		if(i==result.size()-1)
			output<<result[i];
		else
			output<<result[i]<<" ";
#ifdef DEBUG
		std::cout<<result[i]<<" ";
#endif
	}
	output.close();
}

//Run MAP with LSA-TR 
void runLSATR(GraphicalModelType* gm,std::string ofname)
{
	LSATR lsatr(*gm);
	enum DISTANCE {HAMMING, EUCLIDEAN};
	std::cout<<"Running Inference"<<std::endl;
	lsatr.infer();
	std::cout << "MAP Result: " << lsatr.value() << " Bound: "<<lsatr.bound()<<std::endl;
	std::vector<LabelType> result;
	lsatr.arg(result);
	//Write Result
	writeResult(result,ofname);

}
//Run MAP with TRW
void runTRWS(GraphicalModelType* gm,std::string ofname)
{
	TRWS trws(*gm);
	std::cout<<"Running Inference"<<std::endl;
	trws.infer();
	std::cout << "MAP Result: " << trws.value() << " Bound: "<<trws.bound()<<std::endl;
	std::vector<LabelType> result;
	trws.arg(result);
	//Write Result
	writeResult(result,ofname);

}

//Run MAP with Dual Decomposition
void runDualDecomposition(GraphicalModelType* gm,std::string ofname)
{
	DualDecompositionSubGradient ddsg(*gm);
	std::cout<<"Running Inference"<<std::endl;
	ddsg.infer();
	std::cout << "MAP Result: " << ddsg.value() << " Bound: "<<ddsg.bound()<<std::endl;
	std::vector<LabelType> result;
	ddsg.arg(result);
	//Write Result
	writeResult(result,ofname);
}

//Run MAP inference with QPBO
void runQPBO(GraphicalModelType* gm,std::string ofname)
{
	QPBO qpbo(*gm);
	std::cout<<"Running Inference"<<std::endl;
	qpbo.infer();
	std::cout << "MAP Result: " << qpbo.value() << " Bound: "<<qpbo.bound()<<std::endl;
	std::vector<LabelType> result;
	qpbo.arg(result);
	//Write Result
	writeResult(result,ofname);
}

//Run inference with Lazy Flipper initialized to setting in initlab
void runICM(GraphicalModelType* gm,std::string ofname, std::string initlab)
{
	//Initial Labelling
	std::vector<LabelType> startPoint(gm->numberOfVariables());
	std::ifstream input;
	input.open(initlab.c_str());
	int label;
	for(size_t i=0;i<gm->numberOfVariables();i++)
	{
		input>>label;	
		startPoint[i] = LabelType(label);	
#ifdef DEBUG
		std::cout<<"L"<<i<<":"<<label<<" ";
#endif
	}
	//Run Inference
	ICM::Parameter para(startPoint); 
	ICM icm(*gm,para);
	std::cout<<"Running Inference"<<std::endl;
	icm.infer();
	std::cout << "MAP Result: " << icm.value() << " Bound: "<<icm.bound()<<std::endl;
	std::vector<LabelType> result;
	icm.arg(result);
	//Write Result
	writeResult(result,ofname);
}
//Run inference with Lazy Flipper initialized to setting in initlab
void runLazyFlipper(GraphicalModelType* gm,std::string ofname, std::string initlab, size_t subGraphsize)
{
	//Initial Labelling
	std::vector<LabelType> startPoint(gm->numberOfVariables());
	std::ifstream input;
	input.open(initlab.c_str());
	int label;
	for(size_t i=0;i<gm->numberOfVariables();i++)
	{
		input>>label;	
		startPoint[i] = LabelType(label);	
#ifdef DEBUG
		std::cout<<"L"<<i<<":"<<label<<" ";
#endif
	}

	//Run Inference
	size_t maxSubgraphsize;
	if(subGraphsize<=0)
		maxSubgraphsize = gm->numberOfVariables()/2;
	else
		maxSubgraphsize = subGraphsize;
	LazyFlipper::Parameter para(maxSubgraphsize);
	LazyFlipper lf(*gm,para);
	lf.setStartingPoint(startPoint.begin());
	
	std::cout<<"Running Inference"<<std::endl;
	lf.infer();
	std::cout << "MAP Result: " << lf.value() << " Bound: "<<lf.bound()<<std::endl;
	std::vector<LabelType> result;
	lf.arg(result);
	//Write Result
	writeResult(result,ofname);
}



int main(int argc,char* argv[])
{
	std::cout<<"# Arguments: "<<argc<<std::endl;
	if(argc<4 || argc>6){
		std::cout<<"Usage: ./map_solver <input gm file> <output file> <map_solver> <init> <lf_subgraphsize>"<<std::endl;
		std::cout<<"Input in gm format, output file written as 0 1 0 1 .."<<std::endl;
		std::cout<<"map_solver = icm/lazyflipper/lsatr/qpbo/trws/dualdecomposition"<<std::endl;
		std::cout<<"For map_solver=lazyflipper/icm, init is the intialization file"<<std::endl;
		return -1;
	}
	std::string	inputfname = std::string(argv[1]);
	std::string outputfname= std::string(argv[2]);
	std::string mapsolver  = std::string(argv[3]);
	std::string lf_init;
	size_t lf_size=1;
	if(argc>=5)
		lf_init= std::string(argv[4]);
	if(argc==6)
		lf_size= atoi(argv[5]);
	GraphicalModelType gm;
	int index = inputfname.find("uai");
	if(index!=std::string::npos && index==inputfname.size()-3)
	{
		std::cout<<"Loading UAI.."<<std::endl;
		index = readUAI(gm,inputfname.c_str());
		std::cout<<"Done.."<<std::endl;
		if(index!=0)
		{
			return index;
		}
	}
	else
	{
		std::cout<<"Loading GM.."<<std::endl;
		opengm::hdf5::load(gm,inputfname,"gm");
		std::cout<<"Done.."<<std::endl;
	}
	std::cout<<gm.numberOfVariables()<<" "<<gm.numberOfFactors()<<std::endl;	

	//Select MAP solver to run
	if(mapsolver.compare("lazyflipper")==0)
	{	
		std::cout<<"Running LazyFlipper"<<std::endl;
		runLazyFlipper(&gm,outputfname,lf_init,lf_size);
	}
	if(mapsolver.compare("qpbo")==0)
	{
		std::cout<<"Running QPBO"<<std::endl;
		runQPBO(&gm,outputfname);
	}
	if(mapsolver.compare("trws")==0)
	{
		std::cout<<"Runnign TRWS"<<std::endl;
		runTRWS(&gm,outputfname);
	}
	if(mapsolver.compare("dualdecomposition")==0)
	{
		std::cout<<"Runnign Dual Decomposition"<<std::endl;
		runDualDecomposition(&gm,outputfname);
	}
	if(mapsolver.compare("icm")==0)
	{
		std::cout<<"Runnign ICM"<<std::endl;
		runICM(&gm,outputfname,lf_init);
	}
	if(mapsolver.compare("lsatr")==0)
	{
		std::cout<<"Runnign LSATR"<<std::endl;
		runLSATR(&gm,outputfname);
	}
}

