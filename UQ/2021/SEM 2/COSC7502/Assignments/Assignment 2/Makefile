# set some defaults.  ?= means don't override the value if it was set already
MPICXX?=mpic++
CXX?=g++
CXXFLAGS?=-std=c++11 -O2
NVCC?=nvcc
NVFLAGS?=-std=c++11 -O2 --gpu-architecture=sm_35 -Wno-deprecated-gpu-targets

# all targets
TARGETS = a2_serial a2_openmp a2_avx a2_cuda a2_mpi

# The first rule in the Makefile is the default target that is made if 'make' is invoked with
# no parameters.  'all' is a dummy target that will make everything
default : all

## Dependencies

# all targets depend on the helper programs
$(TARGETS) : randutil.h randutil.ipp randutil.o eigensolver.h eigensolver.o

LIBS_a2_serial = -larpack
LIBS_a2_openmp = -larpack
LIBS_a2_avx    = -larpack
LIBS_a2_cuda   = -larpack
LIBS_a2_mpi    = -larpack

CXXFLAGS_a2_openmp = -fopenmp
CXXFLAGS_a2_avx = -mavx -mfma

randutil.o : randutil.h randutil.ipp
eigensolver.o : eigensolver.h

# wildcard rules
%_mpi.o : %_mpi.cpp
	$(MPICXX) $(CXXFLAGS) $(CFLAGS_$(basename $<)) -c $< -o $@

%_mpi : %_mpi.cpp
	$(MPICXX) $(CXXFLAGS) $(CXXFLAGS_$@) $(filter %.o %.cpp, $^) $(LDFLAGS) $(LIBS_$@) $(LIB) -o $@

%.o : %.cu
	$(NVCC) $(NVFLAGS) $(NVFLAGS_$(basename $<)) -c $< -o $@

% : %.cu
	$(NVCC) $(NVFLAGS) $(NVFLAGS_$@) $(filter %.o %.cu, $^) $(LDFLAGS) $(LIBS_$@) $(LIB) -o $@

%.o : %.cpp
	$(CXX) $(CXXFLAGS) $(CFLAGS_$(basename $<)) -c $< -o $@

% : %.cpp
	$(CXX) $(CXXFLAGS) $(CXXFLAGS_$@) $(filter %.o %.cpp, $^) $(LDFLAGS) $(LIBS_$@) $(LIB) -o $@

all : $(TARGETS)

clean:
	rm -f $(TARGETS) *.o

.PHONY: clean default all
