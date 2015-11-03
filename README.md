# fw-inference
This contains code corresponding to the paper
```
Barrier Frank-Wolfe for Marginal Inference
R. Krishnan, S. Lacoste-Julien, D. Sontag
NIPS 2015
```
## Requirements
The code is built to run in python2.7 but interfaces with various open-source packages and libraries. 
Numpy and Scipy are used for the matrix computations. 

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
This code involves several moving parts and cumbersome. 
Ideally, the code would best be implemented in OpenGM's framework using Gurobi for LP/ILP where applicable. 

Create a folder with the name of the experiment
Create a subfolder with 

## Test Cases
All instances that require approximate MAP will require OpenGM to be installed.

Synthetic Results
The github repository contains the bare minimum required for each of the Synthetic files.

If you want to download all the Synthetic test cases without having to run them, 
use the following link and run uncompress.sh
https://www.dropbox.com/s/8z5axf978hsczmx/BarrierFW_NIPS15_TestCases.tar.bz2?dl=0

Unfortunately the UAI files containing the models for the ChineseCharacters are too large (~2G) for github
and so the shell directories have been made available here. 

There are detailed instructions on how to run and get the results using approximate MAP in folder's README file.


## References
- [libDAI](https://staff.fnwi.uva.nl/j.m.mooij/libDAI/)
- [OpenGM](http://hci.iwr.uni-heidelberg.de/opengm2/)
- [toulbar2](https://mulcyber.toulouse.inra.fr/)
