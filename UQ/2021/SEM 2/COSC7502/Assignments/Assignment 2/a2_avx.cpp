// COSC3500, Semester 2, 2021
// Assignment 2
// Main file - AVX version
// Submitted by - Joel Thomas

#include "eigensolver.h"
#include "randutil.h"
#include <string>
#include <iostream>
#include <iomanip>
// Include library to be able to use AVX intrinsics
#include <immintrin.h>

// Variable used for enabling AVX memory to be aligned to memory boundaries
#define ALIGN 32  // Aligned to 256-bit boundary

// Global variables to store matrix M of dimension N x N
double* M = nullptr;
int N = 0;

// AVX implementation of the matrix-vector multiply function
void MatrixVectorMultiply(double* Y, const double* X)
{
   // Use vectors of four 64-bit doubles (256-bit block) since Y, X and M store doubles.
   // x and m store four sequential elements from X and M respectively.
   // y in turn incrementally sums m .* x (.* == element-wise multiplication) to process Y_i in blocks of four
   __m256d y, x, m;
   for (int i = 0; i < N; ++i)
   {
      // Initially set y = [0.0, 0.0, 0.0, 0.0]
      y = _mm256_set1_pd(0.0);
      // Process Y_i in blocks of 4 using y
      for (int j = 0; j < N; j += 4)
      {
         // Setup x and m with next four elements from X and M respectively
         x = _mm256_set_pd(X[j + 3], X[j + 2], X[j + 1], X[j]);
         m = _mm256_set_pd(M[i*N + j + 3], M[i*N + j + 2], M[i*N + j + 1], M[i*N + j]);
         // Use fused multiply-add  to calculate y = m .* x + y (.* == element-wise multiplication)
         y = _mm256_fmadd_pd(m, x, y);
      }

      // Same as y[0] + y[1] + y[2] + y[3] but using AVX intrinsics instead
      y = _mm256_hadd_pd(y, y);
      y = _mm256_add_pd(y, _mm256_permute2f128_pd(y, y, 1));
      // At the end, each element of y will contain Sum_j[M_(i, j)*X_j]
      Y[i] = y[0];

      // Add any remaining excluded products from Sum_j[M_(i, j)*X_j] due to last iteration where j += 4 becomes > N
      int rem = 4*(N/4);
      for (int j = rem; j < N; j++)
      {
         Y[i] += M[i*N + j] * X[j];
      }
   }
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
   N = std::stoi(argv[1]);

   // Allocate memory for the matrix
   M = static_cast<double*>(malloc(N*N*sizeof(double)));

   // Seed the random number generator to a known state
   randutil::seed(42);  // The standard random number.  https://xkcd.com/221/

   // Initialize the matrix.  This is a matrix from a Gaussian Orthogonal Ensemble.
   // The matrix is symmetric.
   // The diagonal entries are gaussian distributed with variance 2.
   // The off-diagonal entries are gaussian distributed with variance 1.
   for (int i = 0; i < N; ++i)
   {
      M[i*N + i] = std::sqrt(2.0) * randutil::randn();
      for (int j = i + 1; j < N; ++j)
      {
         M[i*N + j] = M[j*N + i] = randutil::randn();
      }
   }
   auto FinishInitialization = std::chrono::high_resolution_clock::now();

   // Call the eigensolver
   EigensolverInfo Info = eigenvalues_arpack(N, 100);

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

   // Free memory
   free(M);
}
