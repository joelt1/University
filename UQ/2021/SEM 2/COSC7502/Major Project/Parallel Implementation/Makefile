project:
	@echo "Compiling project.cu..."
	nvcc -O2 -std=c++11 --gpu-architecture=compute_70 -lcurand -lcusolver -lcublas project.cu -o project

clean:
	@echo "Cleaning up..."
	rm -f project
