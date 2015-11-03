/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


#include <dai/trwbp.h>
#include <cmath>

#define DAI_TRWBP_FAST 1
namespace dai {
using namespace std;
void TRWBP::setProperties( const PropertySet &opts ) {
    BP::setProperties( opts );

    if( opts.hasKey("nrtrees") )
        nrtrees = opts.getStringAs<size_t>("nrtrees");
    else
        nrtrees = 1000;
    if( opts.hasKey("rho_its") )
        rho_its = opts.getStringAs<size_t>("rho_its");
    else
        rho_its= 1;
}


PropertySet TRWBP::getProperties() const {
    PropertySet opts = BP::getProperties();
    opts.set( "nrtrees", nrtrees );
    opts.set( "rho_its", rho_its );
    return opts;
}


string TRWBP::printProperties() const {
    stringstream s( stringstream::out );
    string sbp = BP::printProperties();
    s << sbp.substr( 0, sbp.size() - 1 );
    s << ",";
    s << "nrtrees=" << nrtrees << "]";
    s << ",";
    s << "rho_its=" << rho_its << "]";
    return s.str();
}


// This code has been copied from bp.cpp, except where comments indicate TRWBP-specific behaviour
Real TRWBP::logZ() const {
    Real sum = 0.0;
    for( size_t I = 0; I < nrFactors(); I++ ) {
        sum += (beliefF(I) * factor(I).log(true)).sum();  // TRWBP/FBP
        sum += Weight(I) * beliefF(I).entropy();  // TRWBP/FBP
    }
    for( size_t i = 0; i < nrVars(); ++i ) {
        Real c_i = 0.0;
        bforeach( const Neighbor &I, nbV(i) )
            c_i += Weight(I);
        if( c_i != 1.0 )
            sum += (1.0 - c_i) * beliefV(i).entropy();  // TRWBP/FBP
    }
    return sum;
}


// This code has been copied from bp.cpp, except where comments indicate TRWBP-specific behaviour
Prob TRWBP::calcIncomingMessageProduct( size_t I, bool without_i, size_t i ) const {
    Real c_I = Weight(I); // TRWBP: c_I

    Factor Fprod( factor(I) );
    Prob &prod = Fprod.p();
    if( props.logdomain ) {
        prod.takeLog();
        prod /= c_I; // TRWBP
    } else
        prod ^= (1.0 / c_I); // TRWBP
	
    // Calculate product of incoming messages and factor I
    bforeach( const Neighbor &j, nbF(I) )
        if( !(without_i && (j == i)) ) {
            const Var &v_j = var(j);
            // prod_j will be the product of messages coming into j
            // TRWBP: corresponds to messages n_jI
            Prob prod_j( v_j.states(), props.logdomain ? 0.0 : 1.0 );
            bforeach( const Neighbor &J, nbV(j) ) {
                Real c_J = Weight(J);  // TRWBP
                if( J != I ) { // for all J in nb(j) \ I
                    if( props.logdomain )
                        prod_j += message( j, J.iter ) * c_J;
                    else
                        prod_j *= message( j, J.iter ) ^ c_J;
                } else if( c_J != 1.0 ) { // TRWBP: multiply by m_Ij^(c_I-1)
                    if( props.logdomain )
                        prod_j += message( j, J.iter ) * (c_J - 1.0);
                    else
                        prod_j *= message( j, J.iter ) ^ (c_J - 1.0);
                }
            }
            // multiply prod with prod_j
            if( !DAI_TRWBP_FAST ) {
                // UNOPTIMIZED (SIMPLE TO READ, BUT SLOW) VERSION
                if( props.logdomain )
                    Fprod += Factor( v_j, prod_j );
                else
                    Fprod *= Factor( v_j, prod_j );
            } else {
                // OPTIMIZED VERSION
                size_t _I = j.dual;
                // ind is the precalculated IndexFor(j,I) i.e. to x_I == k corresponds x_j == ind[k]
                const ind_t &ind = index(j, _I);

                for( size_t r = 0; r < prod.size(); ++r )
                    if( props.logdomain )
                        prod.set( r, prod[r] + prod_j[ind[r]] );
                    else
                        prod.set( r, prod[r] * prod_j[ind[r]] );
            }
        }
    
    return prod;
}


// This code has been copied from bp.cpp, except where comments indicate TRWBP-specific behaviour
void TRWBP::calcBeliefV( size_t i, Prob &p ) const {
    p = Prob( var(i).states(), props.logdomain ? 0.0 : 1.0 );
    bforeach( const Neighbor &I, nbV(i) ) {
        Real c_I = Weight(I);
        if( props.logdomain )
            p += newMessage( i, I.iter ) * c_I;
        else
            p *= newMessage( i, I.iter ) ^ c_I;
    }
}


void TRWBP::construct() {
    BP::construct();
    _weight.resize( nrFactors(), 1.0 );
    sampleWeights( nrtrees );
    if( props.verbose >= 2 )
        cerr << "Weights: " << _weight << endl;
}


void TRWBP::addTreeToWeights( const RootedTree &tree ) {
    for( RootedTree::const_iterator e = tree.begin(); e != tree.end(); e++ ) {
        VarSet ij( var(e->first), var(e->second) );
        size_t I = findFactor( ij );
        _weight[I] += 1.0;
    }
}


void TRWBP::sampleWeights( size_t nrTrees ) {
    if( !nrTrees )
        return;

    // initialize weights to zero
    fill( _weight.begin(), _weight.end(), 0.0 );

    // construct Markov adjacency graph, with edges weighted with
    // random weights drawn from the uniform distribution on the interval [0,1]
    WeightedGraph<Real> wg;
    for( size_t i = 0; i < nrVars(); ++i ) {
        const Var &v_i = var(i);
        VarSet di = delta(i);
        for( VarSet::const_iterator j = di.begin(); j != di.end(); j++ )
            if( v_i < *j )
                wg[UEdge(i,findVar(*j))] = rnd_uniform();
    }

    // now repeatedly change the random weights, find the minimal spanning tree, and add it to the weights
    for( size_t nr = 0; nr < nrTrees; nr++ ) {
        // find minimal spanning tree
        RootedTree randTree = MinSpanningTree( wg, true );
        // add it to the weights
        addTreeToWeights( randTree );
        // resample weights of the graph
        for( WeightedGraph<Real>::iterator e = wg.begin(); e != wg.end(); e++ )
            e->second = rnd_uniform();
    }

    // normalize the weights and set the single-variable weights to 1.0
    for( size_t I = 0; I < nrFactors(); I++ ) {
        size_t sizeI = factor(I).vars().size();
        if( sizeI == 1 )
            _weight[I] = 1.0;
        else if( sizeI == 2 )
            _weight[I] /= nrTrees;
        else
            DAI_THROW(NOT_IMPLEMENTED);
    }
//	Real hardcode[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.0185, 0.4208, 0.0195, 0.02, 0.4201, 0.3219, 0.0176, 0.0201, 0.2188, 0.4185, 0.4209, 0.2199, 0.2197, 0.4226, 0.0212, 0.3176, 0.2176, 0.0182, 0.0205, 0.0191, 0.2213, 0.422, 0.2218, 0.3223, 0.2204, 0.3216, 0.0195, 0.0211, 0.0206, 0.0199, 0.3178, 0.3182, 0.2193, 0.6207, 0.421, 0.0188, 0.3204, 0.3199, 0.0183, 0.0206, 0.4219, 0.021, 0.018, 0.32, 0.0205};
//	for( size_t I = 0; I < nrFactors(); I++ ) {
//		_weight[I] = hardcode[I];
//		cout<<I<<_weight[I]<<endl;
// }
}


void TRWBP::updateEdgeWeights( Real alpha , string update) {
    //Take a Frank-Wolfe step (alpha) by moving towards MST entropy
	//cout<<" Updating with alpha = "<<alpha<<endl;
	//Setup Gradient vector
	std::vector<Real> gradient;
	gradient.resize(nrFactors());
	fill(gradient.begin(), gradient.end(), 0.0 );
    WeightedGraph<Real> wg;
    for( size_t i = 0; i < nrVars(); ++i ) {
        const Var &v_i = var(i);
        VarSet di = delta(i);
        for( VarSet::const_iterator j = di.begin(); j != di.end(); j++ )
            if( v_i < *j ){
                VarSet ij( v_i, var(findVar(*j)) );
                size_t I = findFactor( ij );
                //Set the weight of the edge to be the -ve mutual information
				//cout<<v_i<<" "<<*j<<beliefV(i)<<" "<<beliefV(findVar(*j))<<" "<<beliefF(I)<<MutualInfo(beliefF(I))<<endl;
                wg[UEdge(i,findVar(*j))] = MutualInfo(beliefF(I));
				gradient[I] = -1*MutualInfo(beliefF(I));
				//cout<<i<<findVar(*j)<<MutualInfo(beliefF(I))<<endl;
            }
    }
	//Check that wg only has added egdes
	//for( WeightedGraph<Real>::iterator e = wg.begin(); e != wg.end(); e++ ) //DEBUG
	//	cout<<e->first<<"-"<<e->second<<endl; //DEBUG
    //Compute MST 
    RootedTree mst = MaxSpanningTree( wg,true);
	if(update.compare("fw")==0){
		cout<<"Updating with alpha = "<<alpha<<" type: "<<update<<endl;
		//Perform Frank-Wolfe step
		//Shrink all the iterates
    	for( size_t I = 0; I < nrFactors(); I++ ) {
    	    size_t sizeI = factor(I).vars().size();
    	    if( sizeI == 2 )//Only for edges
    	        _weight[I] = (1-alpha)*_weight[I];
    	}
		//Expand the iterates you're moving towards
    	for( RootedTree::const_iterator e = mst.begin(); e != mst.end(); e++ ) {
    	    VarSet ij( var(e->first), var(e->second) );
    	    size_t I = findFactor( ij );
    	    _weight[I] = _weight[I] + alpha;
    	}
	}	
	else if(update.compare("linesearch")==0){
		lineSearch(mst,gradient);
	}
	else{
		DAI_THROW(NOT_IMPLEMENTED);
	}
	//Debug:
	/*
	for( RootedTree::const_iterator e = mst.begin(); e != mst.end(); e++ ) //DEBUG
		cout<<e->first<<"-"<<e->second<<endl; //DEBUG
    for( size_t I = 0; I < nrFactors(); I++ ) {
        size_t sizeI = factor(I).vars().size();
        if( sizeI == 2 )//Only for edges
			cout<<beliefF(I)<<endl;
    }
	cout<<"Modified weights"<<_weight<<" with alpha = "<<alpha<<endl;
	*/
}

void TRWBP::lineSearch(const RootedTree &mst, const std::vector<Real> &gradient) {
	//1) Save _weight, setup vector for MST, slope along search direction 
	std::vector<Real> savedWeight;
    savedWeight.resize( nrFactors(), 1.0 );
	for (size_t I =0;I< nrFactors();I++){
		savedWeight[I] = _weight[I];
	}
	std::vector<Real> vertex;
    vertex.resize( nrFactors(), 1.0 );
	fill(vertex.begin(),vertex.end(), 0.0 );
    for( RootedTree::const_iterator e = mst.begin(); e != mst.end(); e++ ) {
        VarSet ij( var(e->first), var(e->second) );
   	    size_t I = findFactor( ij );
		vertex[I] = 1;
   	}
	Real m = 0;
	for (size_t I =0;I< nrFactors();I++){
        size_t sizeI = factor(I).vars().size();
		if( sizeI == 1){}
		else if( sizeI == 2 ){
			m = m+ (vertex[I]-savedWeight[I])*gradient[I];
		}
		else{ DAI_THROW(NOT_IMPLEMENTED);}
		savedWeight[I] = _weight[I];
	}
	//2) Backtracking line search
	Real alpha = 0.9; 
	const Real c = 0.8;
	const Real tau = 0.5;
	Real f_x = runInference(1);
	cout<<"\tStaring line search. LogZ: "<<f_x<<" c="<<c<<" tau="<<tau<<" alpha="<<alpha<<" m="<<m<<endl;
	while(1){
		//a) Update _weight based on alpha
		for (size_t I =0;I< nrFactors();I++){
        	size_t sizeI = factor(I).vars().size();
			if( sizeI == 1){}
			else if( sizeI == 2 ){
				_weight[I] = savedWeight[I] + alpha*(vertex[I]-savedWeight[I]); 
			}
			else{ DAI_THROW(NOT_IMPLEMENTED);}
		}
		BP::init();
		Real f_xnext = runInference(1);
		cout<<"\tf_next: "<<f_xnext<<" f(x+alpha*p)-f(x)="<<f_xnext-f_x<<" alpha*c*m="<<alpha*c*m<<endl;
		if (f_xnext-f_x<=alpha*c*m){
			break;
		}
		else{
			alpha = alpha*tau;
			cout<<"\t Updating alpha to"<<alpha<<endl;
		}
		if (alpha<1e-6){
			cout<<"\t Tolerance Reached: "<<alpha<<endl;
			break;
		}
	}
	cout<<"\tDone line search. alpha* = "<<alpha<<endl;
	//3) Setup _weight with the best value of alpha
	for (size_t I =0;I< nrFactors();I++){
       	size_t sizeI = factor(I).vars().size();
		if( sizeI == 1){}
		else if( sizeI == 2 ){
			_weight[I] = savedWeight[I] + alpha*(vertex[I]-savedWeight[I]); 
		}
		else{ DAI_THROW(NOT_IMPLEMENTED);}
	}
}

Real TRWBP::run() {
    if( props.verbose >= 1 )
        cerr << "Starting " << identify() << "...";
    if( props.verbose >= 3)
        cerr << endl;
	cout<<"Running "<<rho_its<<" tightening steps"<<endl;
	std::vector<Real> savedWeight;
    savedWeight.resize( nrFactors(), 1.0 );
    for (size_t rho_it=1;rho_it<=rho_its;rho_it++) {
        // do several passes over the network until maximum number of iterations has
        // been reached or until the maximum belief difference is smaller than tolerance
		cout<<"["<<rho_it<< "] "<<" Running Marginal Inference"<<endl;
		BP::init();
       	Real lz = runInference(); 
        // Update edge weights 
		cout<<"["<<rho_it<< "] ";
		cout <<"Done. Stats: ";
		//If logZ is not valid, go back to previous set of weights, run inference and quit
		cout<<"It: "<<_iters<<" Maxdiff: "<<_maxdiff<<" LogZ: "<<lz<<endl;
		if (lz!=lz)
		{
			cout<<"LogZ was nan..reverting to weights of iteration: "<<rho_it-1<<endl;
			//Copy over old weights into _weight
    		for( size_t I = 0; I < nrFactors(); I++ ) {
            	_weight[I] = savedWeight[I];
    		}
			rho_it= rho_its-1;
			BP::init();
			BP::construct();
			continue;
		}
		//Track most recent valid weights
    	for( size_t I = 0; I < nrFactors(); I++ ) {
           	savedWeight[I] = _weight[I];
    	}
		if(rho_it<rho_its)
		{
        	//updateEdgeWeights(float(2)/(rho_it+2),string("fw"));
        	updateEdgeWeights(float(2)/(rho_it+2),string("linesearch"));
		}

    }
    return _maxdiff;
}

Real TRWBP::runInference(size_t linesearch) {
	Real maxDiff = INFINITY;
    double tic = toc();
    for( ; _iters < props.maxiter && maxDiff > props.tol && (toc() - tic) < props.maxtime; _iters++ ) {
        if( props.updates == Properties::UpdateType::SEQMAX ) {
            if( _iters == 0 ) {
                // do the first pass
                for( size_t i = 0; i < nrVars(); ++i )
                  bforeach( const Neighbor &I, nbV(i) )
                      calcNewMessage( i, I.iter );
            }
            // Maximum-Residual BP [\ref EMK06]
            for( size_t t = 0; t < _updateSeq.size(); ++t ) {
                // update the message with the largest residual
                size_t i, _I;
                findMaxResidual( i, _I );
                updateMessage( i, _I );
                // I->i has been updated, which means that residuals for all
                // J->j with J in nb[i]\I and j in nb[J]\i have to be updated
                bforeach( const Neighbor &J, nbV(i) ) {
                    if( J.iter != _I ) {
                        bforeach( const Neighbor &j, nbF(J) ) {
                            size_t _J = j.dual;
                            if( j != i )
                                calcNewMessage( j, _J );
                        }
                    }
                }
             }
        } else if( props.updates == Properties::UpdateType::PARALL ) {
            // Parallel updates
            for( size_t i = 0; i < nrVars(); ++i )
                bforeach( const Neighbor &I, nbV(i) )
                    calcNewMessage( i, I.iter );
            for( size_t i = 0; i < nrVars(); ++i )
                 bforeach( const Neighbor &I, nbV(i) )
                    updateMessage( i, I.iter );
        } else {
            // Sequential updates
            if( props.updates == Properties::UpdateType::SEQRND )
               random_shuffle( _updateSeq.begin(), _updateSeq.end(), rnd );

            bforeach( const Edge &e, _updateSeq ) {
                calcNewMessage( e.first, e.second );
                updateMessage( e.first, e.second );
            }
        }
        // calculate new beliefs and compare with old ones
        maxDiff = -INFINITY;
        for( size_t i = 0; i < nrVars(); ++i ) {
            Factor b( beliefV(i) );
            maxDiff = std::max( maxDiff, dist( b, _oldBeliefsV[i], DISTLINF ) );
            _oldBeliefsV[i] = b;
        }
        for( size_t I = 0; I < nrFactors(); ++I ) {
            Factor b( beliefF(I) );
            maxDiff = std::max( maxDiff, dist( b, _oldBeliefsF[I], DISTLINF ) );
            _oldBeliefsF[I] = b;
        }
        if( props.verbose >= 3 )
            cerr << name() << "::run:  maxdiff " << maxDiff << " after " << _iters+1 << " passes" << endl;
    }
    _maxdiff = maxDiff;
	if(linesearch==0)
	{
    	if( props.verbose >= 1 ) {
    	    if( maxDiff > props.tol ) {
    	        if( props.verbose == 1 )
    	            cerr << endl;
    	            cerr << name() << "::run:  WARNING: not converged after " << _iters << " passes (" << toc() - tic << " seconds)...final maxdiff:" << maxDiff << endl;
    	    } else {
    	        if( props.verbose >= 3 )
    	            cerr << name() << "::run:  ";
    	            cerr << "converged in " << _iters << " passes (" << toc() - tic << " seconds)." << endl;
    	    }
    	}
	}
   	return logZ(); 
}

} // end of namespace dai

