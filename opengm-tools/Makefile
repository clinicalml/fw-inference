CC=h5c++
INCLUDES = -I/data/ml2/rahul/software/opengm/opengm-install/include -I/data/ml2/rahul/software/opengm/opengm-master/src/external/QPBO-v1.3.src-patched -I/data/ml2/rahul/software/opengm/opengm-master/src/external/TRWS-v1.3.src-patched -I/data/ml2/rahul/software/opengm/opengm-master/src/external/MaxFlow-v3.02.src-patched -I/data/ml2/rahul/software/vigra/vigra-install/include 
LFLAGS = -L/usr/lib64 -L/data/ml2/rahul/software/opengm/opengm-build/src/external/ 
OPTFLAGS = -O3 
CFLAGS  = -c -g -DUSE_64_BIT_PTR_CAST #-Wall
#Set to -DDEBUG
DEBUG   = -DDEBUG 

all: map_solver

map_solver: map_solver.o read_uai.o 
	$(CC) map_solver.o read_uai.o -o map_solver $(LFLAGS) -lexternal-library-qpbo -lexternal-library-trws -lexternal-library-maxflow 
	#-lexternal-library-mrf

map_solver.o: map_solver.cpp 
	$(CC) $(DEBUG) $(INCLUDES) $(OPTFLAGS) $(CFLAGS) map_solver.cpp 

read_uai.o: read_uai.cpp 
	$(CC) $(DEBUG) $(INCLUDES) $(OPTFLAGS) $(CFLAGS) read_uai.cpp 
clean:
	rm -rf *o map_solver
