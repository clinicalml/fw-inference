CC=g++49
#For TRWBPOPT
INCLUDES_OPT = -I/data/ml2/rahul/software/libDAI_trwbp_opt/include -I/home/rahul/software/gmp-6.0.0/include 
LFLAGS_OPT = -L/usr/lib64 -L/data/ml2/rahul/software/libDAI_trwbp_opt/lib -L/home/rahul/software/gmp-6.0.0/lib
OPTFLAGS = -O3 -fpic
CFLAGS  = -c -g -Wno-deprecated -Wall -W -Wextra -static 
#Set to -DDEBUG
DEBUG   = -DDEBUG 

all: trwbp exact

trwbp: trwbp.o 
	$(CC) trwbp.o -o trwbp $(LFLAGS_OPT) -lgmpxx -lgmp -ldai

trwbp.o: trwbp.cpp 
	$(CC) $(DEBUG) $(INCLUDES_OPT) $(OPTFLAGS) $(CFLAGS) trwbp.cpp 

exact.o: exact.cpp
	$(CC) $(DEBUG) $(INCLUDES_OPT) $(OPTFLAGS) $(CFLAGS) exact.cpp 

exact: exact.o
	$(CC) exact.o -o exact $(LFLAGS_OPT) -lgmpxx -lgmp -ldai

clean:
	rm -rf *o trwbp exact
