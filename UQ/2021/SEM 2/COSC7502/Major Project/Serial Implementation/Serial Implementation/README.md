# README for COSC7502 serial implementation authored by Joel Thomas.
# Explanation of the different files found in this zip folder:
1) The Eigen and Ziggurat folders that are located next to the parent folder of this file are used for the Eigen C++ library and C++ Ziggurat algorithm used in project.cpp.
2) The project.cpp file contains the serial implementation programmed in C++.
3) The run.bat file can be used to compile, execute and clean up an -O2 compiler-optimised version of project.cpp on a Windows PC.
4) A Makefile and Slurm bash script (goslurm.sh) have been provided to perform the same instructions in step 3) but on UQ SMP's Getafix HPC cluster instead.
5) The project_output.txt file contains the standard output one should expect to receive when performing step 4) above.
6) The profiling.txt file contains the gprof utility tool's profiling output when compiled with g++ and with the -pg and -O0 flags enabled.
7) The diagrams.ipynb file is a Jupyter Notebook file that is used to generate figures 3 and 4 in the project report. The required Python libraries are matplotlib, numpy and jupyter and can be installed via the pip package manager for Python.
8) The Tests folder contain five sub-folders, one for each of the compiler optimisations tested as in section 1.3 of the project report. Within each of these sub-folders, a Makefile and Slurm bash script (goslurm.sh) are provided to compile and run the project on the cluster and conduct the benchmark tests. The output from these are represented as N100_M10000.txt files where the N100 and M10000 represent the values of N and M respectively that are used in conducting the test.