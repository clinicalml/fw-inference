//Load LIBDAI GM, run TRBP and output marginals
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <map>
#include <dai/alldai.h>  
#include <dai/trwbp.h>
#include <dai/decmap.h>
#include <dai/io.h>
#include <ctime>
using namespace dai;

int main( int argc, char *argv[] ) {
	std::cout<<"# Arguments: "<<argc<<std::endl;
	long double damping_in=0.1;
	size_t nrtrees = 1000;
	size_t maxits = 1000;
	size_t rho_its= 1;
	if(argc<3 || argc>7){
		std::cout<<"Usage: ./trwbp <input UAI file> <output file basename> <*rho_its=1> <*maxits=10K> <*damping=0.01> <*nrtrees=1K>"<<std::endl;
		std::cout<<"Will write .log file and .margfile, * indicates optional"<<std::endl;
		return 0;
	}
	else {
		std::string suffix = "";
		// Read FactorGraph from the file specified by the first command line argument
		std::string	inputfname = std::string(argv[1]);
		std::string outputfname= std::string(argv[2]);
		if(argc>=4)
		{
			rho_its= std::atol(argv[3]);
			std::cout<<"Modifying rho_its"<<std::endl;
			suffix += "_rho_its_"+std::string(argv[3]);
		}
		if(argc>=5)
		{
			maxits = std::atol(argv[4]);
			std::cout<<"Modifying maxits"<<std::endl;
			suffix += "_maxits_"+std::string(argv[4]);
		}
		if(argc>=6)
		{
			damping_in = std::atof(argv[5]);
			std::cout<<"Modifying damping_in"<<std::endl;
			suffix += "_damping_"+std::string(argv[5]);
		}

		if(argc==7)
		{
			nrtrees = std::atol(argv[6]);
			std::cout<<"Modifying nrtrees"<<std::endl;
			suffix += "_trees_"+std::string(argv[6]);
		}

		std::vector<Var> vars;
		std::vector<Factor> facs0;
		std::vector<Permute> permutations;
		bool verbose = true;
		ReadUaiAieFactorGraphFile(inputfname.c_str(), verbose, vars, facs0, permutations );
		FactorGraph fg( facs0.begin(), facs0.end(), vars.begin(), vars.end(), facs0.size(), vars.size() );
		
		//Start Timer
		std::clock_t start = clock(); 
		//Parameters
		Real   tol = 1e-5;
		PropertySet opts;
		size_t verb = 1;
		Real damping = (Real)damping_in;
		opts.set("maxiter",maxits);  
		opts.set("tol",tol);          
		opts.set("verbose",verb);
		opts.set("updates",std::string("SEQRND"));
		//opts.set("updates",std::string("PARALL"));
		opts.set("logdomain",true);
		opts.set("nrtrees",nrtrees);
		opts.set("damping",damping);
		opts.set("rho_its",rho_its);
		std::cout<<"Running with "<<opts<<std::endl;
		//Run TRWBP
		TRWBP trwbp(fg, opts);
		trwbp.init();
		std::clock_t end_init = clock();
		double elapsed_seconds = double(end_init-start)/CLOCKS_PER_SEC;
		std::cout<<"Time Elapsed INIT: "<<elapsed_seconds<<std::endl;
		trwbp.run();
		//End Timer
		std::clock_t end = clock();
		elapsed_seconds = double(end-end_init)/CLOCKS_PER_SEC;
		std::cout<<"Time Elapsed RUN "<<elapsed_seconds<<std::endl;

		//Write results
		outputfname += suffix;
		std::string logfname = outputfname+(".log");
		outputfname += ".trbp";
		std::ofstream output(outputfname.c_str());
		output<<(trwbp.logZ()/dai::log((Real)dai::exp((Real)1)))<<std::endl;
		output<<elapsed_seconds<<std::endl;
		std::ofstream logout(logfname.c_str());

		logout << "Approximate (TRWBP) log partition sum: " << (trwbp.logZ()/dai::log((Real)dai::exp((Real)1)))<< std::endl;
		logout << "Approximate (TRWBP) variable marginals:" << std::endl;
		for( size_t i = 0; i < fg.nrVars(); i++ )
		{
			output << trwbp.belief(fg.var(i))[0]<<std::endl<<trwbp.belief(fg.var(i))[1] << std::endl; 
			logout << trwbp.belief(fg.var(i)) << std::endl; 
		}
		logout << "Approximate (TRWBP) factor marginals:" << std::endl;
		for( size_t I = 0; I < fg.nrFactors(); I++ ) 
		{
			logout << trwbp.belief(fg.factor(I).vars()) << std::endl; 
		}
		
	}

	return 0;
}
/*
		InfAlg *trwbp_alg =newInfAlgFromString( "TRWBP[inference=SUMPROD,updates=SEQRND,logdomain=1,tol=1e-9,maxiter=10000,damping=0.01,nrtrees=10000,verbose=1]",fg);
		trwbp_alg->init();
		trwbp_alg->run();
*/
