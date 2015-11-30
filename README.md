# fw-inference

NOTE: This repository is still under construction!!!


This contains code corresponding to the paper
```
Barrier Frank-Wolfe for Marginal Inference
R. Krishnan, S. Lacoste-Julien, D. Sontag
NIPS 2015
```
## Requirements
The code is built to run in python2.7 but interfaces with various open-source packages and libraries. 
Numpy and Scipy are used for the matrix computations. 
Gurobi is used to solve Integer Linear Programs (or LPs)

The code also needs the following packages installed:
* [Required: Gurobi - Exact MAP solver](http://www.gurobi.com/)

```
#Check that the following works in the python interpreter:
>>>import gurobipy
>>>
```
* [Optional: Toulbar2 - Exact MAP solver (version>=0.9.7.0)](https://mulcyber.toulouse.inra.fr/projects/toulbar2/)
```
Attached the source for toulbar2.0.9.7.0 
Contains a small modification to the way the output files are handled.
To see modifications, look at lines 1630-1650 in src/tb2main.cpp
```

* [Optional: OpenGM - Approximate MAP solvers](http://hci.iwr.uni-heidelberg.de/opengm2/)
```
This is necessary in order to use approximate MAP solvers. 
1) Copy over the uai2opengm binary to the folder "opengm-tools" 
2) Modify the Makefile to look for headers and libraries as appropriate
3) Run 'make'
```

* [Optional: Patched LibDAI - Comparison with TRWBP](https://staff.fnwi.uva.nl/j.m.mooij/libDAI/)
```
This is necessary for code used for comparisons with TRBP. 
1) Download the source for libDAI
2) IMPORTANT: Replacing the files trwbp.cpp and trwbp.h in libDAIs source code (src & include folder respectively). 
These files contain a wrapper that performs tightening of the optimization over the spanning tree polytope. 
This is not performed by default.
3) Build libDAI normally. 
```


## Instructions
This code involves several moving parts and can be cumbersome. 
Future: The code would best be implemented in OpenGM's framework where LP/ILP/MAP solvers are readily available 

1] Create a folder with the name of the experiment (eg. SyntheticGrids_5x5)
2] Create a subfolder titled models containing .uai and .uai.evid files 
3] In ``experiments", create a config file and run experiment 

## Test Cases
The repository contains the shell files for the Synthetic test cases. To download 
the pre-run Synthetic test cases, see the file getSyntheticPreRun.sh.
To download the shell files for the Chinese Characters, see getChineseChar.sh

All instances that require approximate MAP will require OpenGM to be installed. See opengm-tools for
more details. 

## Plots
After running the experiments, see the notebooks in the ipynb folder about how to get plots in the figure/visualize
chinese character examples

## References
- [libDAI](https://staff.fnwi.uva.nl/j.m.mooij/libDAI/)
- [OpenGM](http://hci.iwr.uni-heidelberg.de/opengm2/)
- [toulbar2](https://mulcyber.toulouse.inra.fr/)
