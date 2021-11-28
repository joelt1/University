nvcc -O2 --gpu-architecture=compute_86 -lcurand -lcusolver -lcublas project.cu -o project.exe
project.exe 8 1000 1000000
rm -f *.exe *.lib *.exp
