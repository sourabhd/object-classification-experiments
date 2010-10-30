OCTAVE_LIB="/usr/lib/octave-3.2.3"
OCTAVE_INCL="/usr/include/octave-3.2.3"
OPENMPI_INC="/usr/include/openmpi/"	
CC=mpicxx
CFLAGS=-Wall --ansi -O3 -fopenmp 
CFLAGS2=-Wall --ansi -O0 -fopenmp 
#CFLAGS=-Wall -Werror --ansi -O3 -fopenmp
#OPT=-ftree-loop-linear -ftree-loop-im -funswitch-loops -funroll-loops -ftree-vectorize -msse3 -march=athlon64 -m64
#OPT=-ftree-loop-linear -ftree-loop-im -funswitch-loops -funroll-loops -ftree-vectorize -msse3 -march=native -m32
#OPT=-ftree-loop-linear -ftree-loop-im -funswitch-loops -funroll-loops -ftree-vectorize -msse3 -march=athlon64 -m64
OPT=-ftree-loop-linear -ftree-loop-im -funswitch-loops -funroll-loops -ftree-vectorize -msse3 -march=native
DEBUG=-ftree-vectorizer-verbose=2 -g -pg -D _DEBUG
DEBUG2=-g -pg -D _DEBUG
LIBS=-L /usr/lib -L /usr/lib64 -L /usr/local/lib -L $(OCTAVE_LIB) -lcv -lcvaux -lml -lhighgui -lm -lgomp -lboost_filesystem -loctave
INCLUDES=-I/usr/local/include/opencv -I . -I $(OCTAVE_INCL) -I $(OPENMPI_INC)
SOURCES1=ParameterExtractor.cpp emd.c GaborFilterModel.cpp GaborFilterModel2.cpp CarClassifier1.cpp CarClassifier2.cpp SPCarFilter.cpp dir.cpp
HEADERS1=ParameterExtractor.h dir.h GaborFilterModel.h sdd_include.h emd.h
OBJS1=ParameterExtractor.o emd.o GaborFilterModel.o GaborFilterModel2.o CarClassifier1.o CarClassifier2.o SPCarFilter.o dir.o
TARGET1=gfmodel
DOPT=
DEPLOYDIR=$(HOME)/SerrePoggioClassifier

################################################################################


all: $(OBJS1) $(TARGET1)

$(TARGET1): $(SOURCES1) $(HEADERS1)
	#$(CC) $(SOURCES1) -o $(TARGET1) $(CFLAGS) $(OPT) $(INCLUDES) $(LIBS) -D _MPI 
	$(CC) $(OBJS1) -o $(TARGET1) $(CFLAGS) $(OPT) $(INCLUDES) $(LIBS) -D _MPI 

debug: dopt $(OBJS1) $(TARGET1)
	#$(CC) $(SOURCES1) -o $(TARGET1) $(CFLAGS) $(OPT) $(DEBUG) $(INCLUDES) $(LIBS) -D _MPI
	$(CC) $(OBJS1) -o  $(DOPT) $(TARGET1) $(CFLAGS2) $(DEBUG2) $(INCLUDES) $(LIBS) -D _MPI

dopt:
	set DOPT=$(DEBUG2)

#sdd_include.o: sdd_include.cpp
#	$(CC) -o sdd_include.o $(DOPT) $(CFLAGS) $(OPT) $(INCLUDES) -D _MPI sdd_include.cpp

ParameterExtractor.o: ParameterExtractor.cpp ParameterExtractor.h
	$(CC) -o ParameterExtractor.o -c $(DOPT) $(CFLAGS) $(OPT) $(INCLUDES) $(LIBS) ParameterExtractor.cpp
emd.o: emd.c emd.h
	gcc -o emd.o -c emd.c
GaborFilterModel.o: GaborFilterModel.cpp GaborFilterModel.h
	$(CC) -o GaborFilterModel.o -c $(DOPT) $(CFLAGS) $(OPT) $(INCLUDES) -D _MPI $(LIBS) GaborFilterModel.cpp
GaborFilterModel2.o: GaborFilterModel2.cpp GaborFilterModel.h
	$(CC) -o GaborFilterModel2.o -c $(DOPT) $(CFLAGS) $(OPT) $(INCLUDES) -D _MPI $(LIBS) GaborFilterModel2.cpp
CarClassifier1.o: CarClassifier1.cpp GaborFilterModel.h
	$(CC) -o CarClassifier1.o -c $(DOPT) $(CFLAGS) $(OPT) $(INCLUDES) -D _MPI $(LIBS) CarClassifier1.cpp
CarClassifier2.o: CarClassifier2.cpp GaborFilterModel.h
	$(CC) -o CarClassifier2.o -c $(DOPT) $(CFLAGS) $(OPT) $(INCLUDES) -D _MPI $(LIBS) CarClassifier2.cpp
SPCarFilter.o: SPCarFilter.cpp GaborFilterModel.h
	$(CC) -o SPCarFilter.o -c $(DOPT) $(CFLAGS) $(OPT) $(INCLUDES) -D _MPI $(LIBS) SPCarFilter.cpp
dir.o: dir.cpp dir.h
	$(CC) -o dir.o -c $(DOPT) $(CFLAGS) $(DOPT) $(INCLUDES) -D _MPI dir.cpp
	

unsupervised:
	mkoctfile createAdjMatrix.cpp
	mkoctfile loadvectors.cpp
	octave ncut.m

clean:
	rm -f $(OBJS1) $(TARGET1) createAdjMatrix.oct loadvectors.oct createAdjMatrix.o loadvectors.o
