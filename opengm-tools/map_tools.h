//Header files
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <algorithm>
#include <sstream>
#include <vector>
#include <cmath>
#include <limits>

//Graphical Model
#include "opengm/graphicalmodel/graphicalmodel_hdf5.hxx"
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx> 
#include <opengm/operations/minimizer.hxx>
#include <opengm/functions/explicit_function.hxx>
typedef opengm::GraphicalModel<double, opengm::Adder> GraphicalModelType;

/*
typedef double ValueType;
typedef size_t IndexType;
typedef size_t LabelType;
typedef opengm::Adder OperatorType;
typedef opengm::Minimizer AccumulatorType;
typedef opengm::DiscreteSpace<IndexType, LabelType> SpaceType;

typedef opengm::meta::TypeListGenerator<
      opengm::ExplicitFunction<ValueType, IndexType, LabelType>
   >::type FunctionTypeList;

typedef opengm::GraphicalModel<
      ValueType,
      OperatorType,
      FunctionTypeList,
      SpaceType
   > GraphicalModelType;
*/
typedef GraphicalModelType::LabelType LabelType;
typedef GraphicalModelType::IndexType IndexType;
typedef GraphicalModelType::ValueType ValueType;


//readUAI header
int readUAI(GraphicalModelType&,std::string uaifile);
