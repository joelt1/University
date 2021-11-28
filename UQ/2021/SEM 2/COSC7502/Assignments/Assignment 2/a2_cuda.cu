// COSC3500, Semester 2, 2021
// Assignment 2
// Main file - CUDA version
// Submitted by - Joel Thomas

#include "eigensolver.h"
#include "randutil.h"
#include <string>
#include <iostream>
#include <iomanip>

// Global variables to store matrix M of dimension N x N on CPU memory
double* M = nullptr;
int* N = nullptr;
// Global variables to store matrix M and vectors Y & X of dimension N on GPU memory
double* Y_device = nullptr;
double* X_device = nullptr;
double* M_device = nullptr;
int* N_device = nullptr;

// Global variables to store number of GPU threads and thread blocks to be utilised
int Threads = 100;
int Blocks;

// Reports any CUDA errors at runtime and quits the program
void checkError(cudaError_t e)
{
   if (e != cudaSuccess)
   {
      std::cerr << "CUDA error: " << int(e) << " : " << cudaGetErrorString(e) << '\n';
      abort();
   }
}

// CUDA implementation of the matrix-vector multiply function on GPU
__global__
void MatrixVectorMultiplyCUDA(double* Y, const double* X, const double* M, const int* N)
{
   // Compute the index i associated with Y_i handled by this thread
   int i = blockIdx.x*blockDim.x + threadIdx.x;

   // Compute Y_i = Sum_j[M_(i, j)*X_j] entirely on this thread
   if (i < *N)
   {
      for (int j = 0; j < *N; j++)
      {
         Y[i] += M[i*(*N) + j] * X[j];
      }
   }
}

// Use as a helper function instead to pass Y and X onto the GPU device
void MatrixVectorMultiply(double* Y, const double* X)
{
   // Set vector Y to consist only of 0s initially
   memset(Y, 0, (*N)*sizeof(double));

   // Copy (initialised) host vectors to GPU memory from CPU memory
   checkError(cudaMemcpy(Y_device, Y, (*N)*sizeof(double), cudaMemcpyHostToDevice));
   checkError(cudaMemcpy(X_device, X, (*N)*sizeof(double), cudaMemcpyHostToDevice));
   cudaDeviceSynchronize();

   // Perform CUDA-parallelised MatrixVectorMultiply
   MatrixVectorMultiplyCUDA<<<Blocks, Threads>>>(Y_device, X_device, M_device, N_device);

   // Copy final Y back to CPU memory from GPU memory
   checkError(cudaMemcpy(Y, Y_device, (*N)*sizeof(double),cudaMemcpyDeviceToHost));
   cudaDeviceSynchronize();
}

int main(int argc, char** argv)
{
   // Get the current time, for benchmarking
   auto StartTime = std::chrono::high_resolution_clock::now();

   // Get the input size from the command line
   if (argc < 2)
   {
      std::cerr << "expected: matrix size <N>\n";
      return 1;
   }
   int temp_N = std::stoi(argv[1]);
   N = &temp_N;

   // Calculate required number of thread blocks based on input N
   Blocks = (*N + Threads - 1)/Threads;

   // Allocate GPU memory for integer variable N (denoting vector size)
   checkError(cudaMalloc(&N_device, sizeof(int)));
   // Copy given value for N to GPU memory from CPU memory
   checkError(cudaMemcpy(N_device, N, sizeof(int), cudaMemcpyHostToDevice));

   // Allocate CPU memory for matrix M
   M = static_cast<double*>(malloc((*N)*(*N)*sizeof(double)));

   // Allocate GPU memory for vectors Y and X and matrix M
   checkError(cudaMalloc(&Y_device, (*N)*sizeof(double)));
   checkError(cudaMalloc(&X_device, (*N)*sizeof(double)));
   checkError(cudaMalloc(&M_device, (*N)*(*N)*sizeof(double)));


   // Seed the random number generator to a known state
   randutil::seed(42);  // The standard random number.  https://xkcd.com/221/

   // Initialize the matrix.  This is a matrix from a Gaussian Orthogonal Ensemble.
   // The matrix is symmetric.
   // The diagonal entries are gaussian distributed with variance 2.
   // The off-diagonal entries are gaussian distributed with variance 1.
   for (int i = 0; i < *N; ++i)
   {
      M[i*(*N) + i] = std::sqrt(2.0) * randutil::randn();
      for (int j = i + 1; j < *N; ++j)
      {
         M[i*(*N) + j] = M[j*(*N) + i] = randutil::randn();
      }
   }
   
   // Copy (initialised) host matrix M to GPU memory from CPU memory
   checkError(cudaMemcpy(M_device, M, (*N)*(*N)*sizeof(double), cudaMemcpyHostToDevice));

   auto FinishInitialization = std::chrono::high_resolution_clock::now();

   // Call the eigensolver
   EigensolverInfo Info = eigenvalues_arpack(*N, 100);

   auto FinishTime = std::chrono::high_resolution_clock::now();

   auto InitializationTime = std::chrono::duration_cast<std::chrono::microseconds>(FinishInitialization - StartTime);
   auto TotalTime = std::chrono::duration_cast<std::chrono::microseconds>(FinishTime - StartTime);

   std::cout << "Obtained " << Info.Eigenvalues.size() << " eigenvalues.\n";
   std::cout << "The largest eigenvalue is: " << std::setw(16) << std::setprecision(12) << Info.Eigenvalues.back() << '\n';
   std::cout << "Total time:                             " << std::setw(12) << TotalTime.count() << " us\n";
   std::cout << "Time spent in initialization:           " << std::setw(12) << InitializationTime.count() << " us\n";
   std::cout << "Time spent in eigensolver:              " << std::setw(12) << Info.TimeInEigensolver.count() << " us\n";
   std::cout << "   Of which the multiply function used: " << std::setw(12) << Info.TimeInMultiply.count() << " us\n";
   std::cout << "   And the eigensolver library used:    " << std::setw(12) << (Info.TimeInEigensolver - Info.TimeInMultiply).count() << " us\n";
   std::cout << "Total serial (initialization + solver): " << std::setw(12) << (TotalTime - Info.TimeInMultiply).count() << " us\n";
   std::cout << "Number of matrix-vector multiplies:     " << std::setw(12) << Info.NumMultiplies << '\n';
   std::cout << "Time per matrix-vector multiplication:  " << std::setw(12) << (Info.TimeInMultiply / Info.NumMultiplies).count() << " us\n";

   // Free CPU and GPU memory
   free(M);
   cudaFree(Y_device);
   cudaFree(X_device);
   cudaFree(M_device);
   cudaFree(N_device);
}
