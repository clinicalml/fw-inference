# fw-inference
This contains code corresponding to the paper
```
Barrier Frank-Wolfe for Marginal Inference
R. Krishnan, S. Lacoste-Julien, D. Sontag
NIPS 2015
```
## Requirements
The code is in python but needs the following packages installed:
* [Gurobi - Exact MAP solver](http://www.gurobi.com/)

```
#Check that the following works in the python interpreter:
>>>import gurobipy
>>>
```
* [Toulbar2 - Exact MAP solver (version>=0.9.7.0)](https://mulcyber.toulouse.inra.fr/projects/toulbar2/)
```
I have attached a binary with the repo. It contains a small modification to the way the output files are handled. 
```


* [OpenGM - Approximate MAP solvers](http://hci.iwr.uni-heidelberg.de/opengm2/)
```
This is necessary in order to use approximate MAP solvers. 
Copy over the uai2opengm binary to the folder "opengm-tools" and run the Makefile after modifying it to point
to the appropriate binaries. 
```

* [Patched LibDAI - Comparison with TRWBP](https://staff.fnwi.uva.nl/j.m.mooij/libDAI/)
```
1) Download the source for libDAI
2) IMPORTANT: Replacing the files trwbp.cpp and trwbp.h in libDAIs source code (src & include folder respectively). These files contain a wrapper that performs tightening of the optimization over the spanning tree polytope. 
3) Build libDAI normally. 
```
The code used for the comparisons with TRBP. 

## Instructions
This code involves several moving parts and cumbersome. Hopefully the following detailed instructions

Create a folder with the name of the experiment
Create a subfolder with 
