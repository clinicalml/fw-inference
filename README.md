# fw-inference
This contains code corresponding to the paper

Barrier Frank-Wolfe for Marginal Inference
R. Krishnan, S. Lacoste-Julien, D. Sontag
NIPS 2015

## Requirements
The code is in python but needs the following packages installed:
* [Gurobi - Exact MAP solver](http://www.gurobi.com/)
Check that the following works:
```
import gurobipy
```
* [Toulbar2 - Exact MAP solver](https://mulcyber.toulouse.inra.fr/projects/toulbar2/)
* [OpenGM - Approximate MAP solvers](http://hci.iwr.uni-heidelberg.de/opengm2/)
* [Sampling Spanning Trees](https://github.com/rahulk90/sample-spanning)
