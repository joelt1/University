/*
Subject - COSC7502
Project Title - HPC in MC Simulation for European Basket Option Pricing
Assessment Title - Parallel Implementation (Part B)
Author - Joel Thomas
*/

// Library/module imports
#include <iostream>								// Input-output streaming
#include <iomanip>								// Input-output manipulation
#include <chrono>								// Manual runtime profiling
#include <ctime>								// Get and manipulate date and time information
#include <cmath>								// Standard C math library incuding functions such as sqrt(), exp(), etc.
#include <cuda_runtime.h>               		// Defines the public host functions and types for the CUDA runtime API
#include <curand.h>                     		// Initialise vectors/matrices with random numbers from a distribution on the device
#include <cusolverDn.h>                 		// Decompositions and linear system solutions for both dense and sparse matrices
#include <cublas_v2.h>                  		// Basic linear algebra subroutine (BLAS)
#include <thrust/device_ptr.h>					// Stores a pointer to an object allocated in device memory
#include <thrust/reduce.h>						// Generalisation of parallel reduction for array summation
#define idx(i, j, lead_dim) (j*lead_dim + i)	// Compute corresponding column-major index for matrix stored on the device
#define THREADS 1024							// Number of GPU threads to use for vectors (ONLY)

// Define variable scope
using namespace std;

// Function headers
void check_general_errors(cudaError_t e, int line);
void check_cuRAND_errors(curandStatus_t e, int line);
void check_cuSOLVER_errors(cusolverStatus_t e, int line);
void check_cholesky_success(int devInfo, int line);
void check_cuBLAS_errors(cublasStatus_t e, int line);
__global__ void fill_vec(const int *d_dim, float *d_vec, float *d_value);
__global__ void fill_mat(const int *d_dim1, const int *d_dim2, float *d_mat, float *d_value);
float* initialise_correl_mat(const int D);
float* perform_cholesky(const int D);
void process_L_mat(const int D, float *L);
__global__ void euler_timestep(const int *d_D, const int *d_M, const float *d_dt, const float *d_sqrt_dt,
		float *d_S, const float *d_mu, const float *d_sigma, const float *d_ZTLT);
__global__ void calc_disc_payoff(const int *d_D, const int *d_M, const int *d_K, const float *d_exp__rt,
		const float *d_S, float *d_Y);
__global__ void calc_sq_dev(const int *d_M, const float *hat_C_M, float *d_Y);
void perform_mc_simulation(const int D, const int N, const int M, const int K, const float r, const int T);
void print_results(const float hat_C_M, const float CI_left, const float CI_right, const float radius);
void free_params(int* d_D, int *d_M, int *d_K, float *d_r, float *d_dt, float *d_sqrt_dt, float *d_exp__rt,
		float *d_value, float *d_hat_C_M);
void free_vecs_mats(float *d_S, float *d_mu, float *d_sigma, float *d_L, float *d_Z, float *d_LZ,
		float *d_ZTLT, float *d_Y, cublasHandle_t handle);

int main(int argc, char *argv[]) {
	/*
	Initialises all simulation variables before passing them onto the main perform_mc_simulation() function.
	*/
	char project_topic[] = "HPC in MC Simulation for European Basket Options Pricing - Parallel Implementation";
	cout << project_topic << endl;
	cout << "Starting program...\n" << endl;

	// General parameters
	const int K = 100;		// Contract strike price
	const float r = 0.02;	// Risk-free interest rate
	const int T = 1;		// Contract time to expiry (>= 0)
	
	// Number of assets to simulate
	int D;
	// Number of desired intermediate time intervals between t=0 and t=T
	int N;
	// Number of MC samples to generate
	int M;

	if (argc > 1) {
		D = atoi(argv[1]);
		N = atoi(argv[2]);
		M = atoi(argv[3]);
	} else {
		D = 8;
		N = 100;
		M = 10000;
	}

	cout << "D = " << D << ", N = " << N << ", M = " << M << endl;

	// Start profiling
	auto start_time = chrono::high_resolution_clock::now();

	// Run main MC simulation function
	perform_mc_simulation(D, N, M, K, r, T);

	// Stop profiling
	auto stop_time = chrono::high_resolution_clock::now();
	auto runtime = chrono::duration_cast<chrono::milliseconds>(stop_time - start_time).count();
	cout << "Total runtime\t\t\t= " << runtime << endl;

	return 0;
}

void check_general_errors(cudaError_t e, int line) {
	/*
	Reports any CUDA errors at runtime and quits the program.
	*/
	if (e != cudaSuccess) {
		cerr << "CUDA Error " << int(e) << " raised on line " << line << ": " << cudaGetErrorString(e) << endl;
		abort();
	}
}

void check_cuRAND_errors(curandStatus_t e, int line) {
	/*
	Reports any cuRAND errors at runtime and quits the program.
	*/
	if (e != CURAND_STATUS_SUCCESS) {
		cerr << "CURAND Error " << int(e) << " raised on line " << line << endl;
		cerr << "CUDA Error: " << cudaGetErrorString(cudaGetLastError()) << endl;
		abort();
	}
}

void check_cuSOLVER_errors(cusolverStatus_t e, int line) {
	/*
	Reports any cuSOLVER errors at runtime and quits the program.
	*/
	if (e != CUSOLVER_STATUS_SUCCESS) {
		cerr << "CUSOLVER Error " << int(e) << " raised on line " << line << endl;
		cerr << "CUDA Error: " << cudaGetErrorString(cudaGetLastError()) << endl;
		abort();
	}
}

void check_cholesky_success(int devInfo, int line) {
	/*
	Verifies the success of Cholesky factorisation of P which should be Hermitian positive-definite
	matrix in order to not raise any errors.
	*/
	if (devInfo > 0) {
		cerr << "Unsuccessful Cholesky factorisation on line " << line << endl;
		cerr << "Leading minor of order " << devInfo << " is not positive definite" << endl;
		abort();
	} else if (devInfo < 0) {
		cerr << "Unsuccessful Cholesky factorisation on line " << line << endl;
		cerr << "Parameter " << devInfo << " is wrong (not counting handle)" << endl;
		abort();
	}
}

void check_cuBLAS_errors(cublasStatus_t e, int line) {
	/*
	Reports any cuBLAS errors at runtime and quits the program.
	*/
	if (e != CUBLAS_STATUS_SUCCESS) {
		cerr << "CUBLAS Error " << int(e) << " raised on line " << line << endl;
		cerr << "CUDA Error: " << cudaGetErrorString(cudaGetLastError()) << endl;
		abort();
	}
}

__global__
void fill_vec(const int *d_dim, float *d_vec, float *d_value) {
	/*
	Fills the vector d_vec, of dimension d_dim x 1, stored on the device with value d_value.
	*/
	const int dim = *d_dim;
	const float value = *d_value;

	// Find the relevant vector index handled by this thread
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < dim) {
		d_vec[i] = value;
	}
}

__global__
void fill_mat(const int *d_dim1, const int *d_dim2, float *d_mat, float *d_value) {
	/*
	Fills the matrix d_mat, of dimension d_dim1 x d_dim2, stored on the device with value d_value.
	*/
	const int dim1 = *d_dim1;
	const int dim2 = *d_dim2;
	const float value = *d_value;

	// Find the relevant matrix indices handled by this thread
	int i = blockIdx.y*blockDim.y + threadIdx.y;
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	

	if (i < dim1 && j < dim2) {
		d_mat[i*dim2 + j] = value;
	}
}

float* initialise_correl_mat(const int D) {
	/*
	Initialises correlation matrix P of dimension D x D as the identity matrix by default i.e. uncorrelated,
	independent assets.
	*/
	float *P = static_cast<float*>(malloc(D*D*sizeof(float)));
	for (int i = 0; i < D; i++) {
		for (int j = 0; j < D; j++) {
			if (i == j) {
				P[idx(i, j, D)] = 1.0f;
			} else {
				P[idx(i, j, D)] = 0.0f;
			}
		}
	}

	return P;
}

float* perform_cholesky(const int D) {
	/*
	Perform the Cholesky factorisation of matrix P and store the result in matrix L of dimension D x D.
	*/
	// Initialise P on the host
	float *P = initialise_correl_mat(D);

	// Create cuSOLVER handle
	cusolverDnHandle_t handle;
	float *d_P;

	// Allocate GPU memory then copy P from host to device
	check_general_errors(cudaMalloc(&d_P, D*D*sizeof(float)), __LINE__);
	check_general_errors(cudaMemcpy(d_P, P, D*D*sizeof(float), cudaMemcpyHostToDevice), __LINE__);

	// Stores whether the Cholesky factorisation of P was successful or not, on the device
	int *devInfo;
	check_general_errors(cudaMalloc(&devInfo, sizeof(int)), __LINE__);

	// Stores the value of the necessary size of work buffers required to Cholesky factorise P
	check_cuSOLVER_errors(cusolverDnCreate(&handle), __LINE__);
	int Lwork = 0;
	check_cuSOLVER_errors(cusolverDnSpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER, D, d_P, D, &Lwork),
		__LINE__);
	check_general_errors(cudaDeviceSynchronize(), __LINE__);

	// Allocate size of work buffers required to Cholesky factorise P
	float *Workspace;
	check_general_errors(cudaMalloc(&Workspace, Lwork*sizeof(float)), __LINE__);

	// Perform Cholesky factorisation of P
	check_cuSOLVER_errors(cusolverDnSpotrf(handle, CUBLAS_FILL_MODE_LOWER, D, d_P, D, Workspace, Lwork,
		devInfo), __LINE__);
	check_general_errors(cudaDeviceSynchronize(), __LINE__);

	// Verify success of Cholesky factorisation of P. Successful only if h_devInfo = 0 after copying memory from
	// devInfo to h_devInfo i.e. device to host
	int h_devInfo = -1;
	check_general_errors(cudaMemcpy(&h_devInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost), __LINE__);
	check_cholesky_success(h_devInfo, __LINE__);

	// If Cholesky factorisation successful, store result in L
	float *L = static_cast<float*>(malloc(D*D*sizeof(float)));
	check_general_errors(cudaMemcpy(L, d_P, D*D*sizeof(float), cudaMemcpyDeviceToHost), __LINE__);
	check_general_errors(cudaDeviceSynchronize(), __LINE__);
	
	// Result of Cholesky factorisation of P only rewrites the lower triangular part of P while leaving the upper
	// triangular part unchanged, so need to manually set upper part of L to 0
	process_L_mat(D, L);

	free(P);
	cusolverDnDestroy(handle);
	cudaFree(d_P);

	return L;
}

void process_L_mat(const int D, float *L) {
	/*
	Manually set upper triangular part of L to 0, read explanation provided on lines 254-255.
	*/ 
	for (int i = 0; i < D; i++) {
		for (int j = 0; j < D; j++) {
			if (i < j) {
				L[idx(i, j, D)] = 0.0;
			}
		}
	}
}

__global__
void euler_timestep(const int *d_D, const int *d_M, const float *d_dt, const float *d_sqrt_dt,
		float *d_S, const float *d_mu, const float *d_sigma, const float *d_ZTLT) {
	/*
	Perform Euler timestepping to advance each simulated sample for each of the D assets from S_(n) at
	timestep n to S_(n+1) at timestep n+1.
	*/
	const int D = *d_D;
	const int M = *d_M;
	const float dt = *d_dt;
	const float sqrt_dt = *d_sqrt_dt;

	// Find the relevant matrix indices handled by this thread
	int m = blockIdx.y*blockDim.y + threadIdx.y;
	int d = blockIdx.x*blockDim.x + threadIdx.x;

	if (m < M && d < D) {
		// Calculate S_(n+1)^(m, d) = max(S_n^(m, d)*(1 + mu^(d)*dt + sigma^(d)*sqrt(dt)*((L*Z_n).T)^(m, d)), 0)
		// d_S[m*D + d] *= 1.0f + d_mu[d]*dt + d_sigma[d]*sqrt_dt*d_ZTLT[m*D + d];
		d_S[m*D + d] *= 1.0f + d_mu[d]*dt + d_sigma[d]*sqrt_dt*d_ZTLT[m*D];
		if (d_S[m*D + d] < 0) {
			d_S[m*D + d] = 0.0f;
		}
	}
}

__global__
void calc_disc_payoff(const int *d_D, const int *d_M, const int *d_K, const float *d_exp__rt,
		const float *d_S, float *d_Y) {
	/*
	Calculate the discounted contract payoff vector Y after simulating M samples for each of the D assets
	till option maturity at time T.
	*/
	const int D = *d_D;
	const int M = *d_M;
	const int K = *d_K;
	const float exp__rt = *d_exp__rt;

	// Find the relevant vector index handled by this thread
	int m = blockIdx.x*blockDim.x + threadIdx.x;

	if (m < M) {
		// Calculate the m-th row-sum (S_(N)^(m, 1) + ... + S_(N)^(m, D))
		for (int d = 0; d < D; d++) {
			d_Y[m] += d_S[m*D + d];
		}

		// Calculate max(1/D * (S_(N)^(m, 1) + ... + S_(N)^(m, D)) - K, 0)
		d_Y[m] = d_Y[m]/D - K;
		if (d_Y[m] < 0) {
			d_Y[m] = 0.0f;
		}

		// Calculate exp(-r*T)*max(1/D * (S_(N)^(m, 1) + ... + S_(N)^(m, D)) - K, 0)
		d_Y[m] *= exp__rt;
	}
}

__global__
void calc_sq_dev(const int *d_M, const float *hat_C_M, float *d_Y) {
	/*
	Calculates and stores squared deviations of elements of d_Y using the precomputed mean of this vector hat_C_M.
	Rewrites (replaces) the m-th discounted payoff of Y with the m-th squared deviation.
	*/
	const int M = *d_M;
	const float mean = *hat_C_M;

	// Find the relevant vector index handled by this thread
	int m = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (m < M) {
		// First calculate the deviation from the mean
		d_Y[m] -= mean;
		// Then calculate the squared deviation
		d_Y[m] *= d_Y[m];
	}
}

void perform_mc_simulation(const int D, const int N, const int M, const int K, const float r, const int T) {
	/*
	Takes as input the key simulation variables initialised in main() and performs the tasks appearing in Algorithm 2
	of the project report. These include initialising the RNG, performing Cholesky decomposition to retrieve L,
	performing MC + Euler Timestepping, calculating the discounted payoff after simulation to time T, generating
	a 95% confidence interval based on the simulation results and finally neatly printing out key results to the console.
	*/
	// Declare pointers to key scalars, vectors and matrices from Algorithm 2 to be stored on device
	int *d_D, *d_M, *d_K;
	d_D = d_M = d_K = nullptr;
	float *d_r, *d_dt, *d_sqrt_dt, *d_exp__rt, *d_S, *d_mu, *d_sigma, *d_L, *d_Z, *d_LZ, *d_ZTLT, *d_Y, \
		*d_value, *d_hat_C_M;
	d_r = d_dt = d_sqrt_dt = d_exp__rt = d_S = d_mu = d_sigma = d_L = d_Z = d_LZ = d_ZTLT = d_Y = \
		d_value = d_hat_C_M = nullptr;
	float dt = (float) T/N;
	float sqrt_dt = sqrt(dt);
	float exp__rt = exp(-r * T);

	// Allocate memory for scalars on device
	check_general_errors(cudaMalloc(&d_D, sizeof(int)), __LINE__);
	check_general_errors(cudaMalloc(&d_M, sizeof(int)), __LINE__);
	check_general_errors(cudaMalloc(&d_K, sizeof(int)), __LINE__);
	check_general_errors(cudaMalloc(&d_r, sizeof(float)), __LINE__);
	check_general_errors(cudaMalloc(&d_dt, sizeof(float)), __LINE__);
	check_general_errors(cudaMalloc(&d_sqrt_dt, sizeof(float)), __LINE__);
	check_general_errors(cudaMalloc(&d_exp__rt, sizeof(float)), __LINE__);
	check_general_errors(cudaMalloc(&d_value, sizeof(float)), __LINE__);
	check_general_errors(cudaMalloc(&d_hat_C_M, sizeof(float)), __LINE__);

	// Initialise scalars on device
	check_general_errors(cudaMemcpyAsync(d_D, &D, sizeof(int), cudaMemcpyHostToDevice), __LINE__);
	check_general_errors(cudaMemcpyAsync(d_M, &M, sizeof(int), cudaMemcpyHostToDevice), __LINE__);
	check_general_errors(cudaMemcpyAsync(d_K, &K, sizeof(int), cudaMemcpyHostToDevice), __LINE__);
	check_general_errors(cudaMemcpyAsync(d_r, &r, sizeof(float), cudaMemcpyHostToDevice), __LINE__);
	check_general_errors(cudaMemcpyAsync(d_dt, &dt, sizeof(float), cudaMemcpyHostToDevice), __LINE__);
	check_general_errors(cudaMemcpyAsync(d_sqrt_dt, &sqrt_dt, sizeof(float), cudaMemcpyHostToDevice), __LINE__);
	check_general_errors(cudaMemcpyAsync(d_exp__rt, &exp__rt, sizeof(float), cudaMemcpyHostToDevice), __LINE__);
	check_general_errors(cudaDeviceSynchronize(), __LINE__);

	// Allocate memory for vectors and matrices on device
	check_general_errors(cudaMalloc(&d_S, M*D*sizeof(float)), __LINE__);
	check_general_errors(cudaMalloc(&d_mu, D*sizeof(float)), __LINE__);
	check_general_errors(cudaMalloc(&d_sigma, D*sizeof(float)), __LINE__);
	check_general_errors(cudaMalloc(&d_L, D*D*sizeof(float)), __LINE__);
	check_general_errors(cudaMalloc(&d_Z, D*M*sizeof(float)), __LINE__);
	check_general_errors(cudaMalloc(&d_LZ, D*M*sizeof(float)), __LINE__);
	check_general_errors(cudaMalloc(&d_ZTLT, M*D*sizeof(float)), __LINE__);
	check_general_errors(cudaMalloc(&d_Y, M*sizeof(float)), __LINE__);

	// Create and assign a 2D CUDA thread grid, see figure 6 in the report
    dim3 dim_block(D, D);  // Total of D*D threads per block
    // Calculate the required number of blocks along X and Y in this 2D CUDA thread grid
    dim3 dim_grid((D + dim_block.x - 1)/dim_block.x, (M + dim_block.y - 1)/dim_block.y);

	// Initialise matrix S_0 directly on device
	float value = 100.0f;
	check_general_errors(cudaMemcpy(d_value, &value, sizeof(float), cudaMemcpyHostToDevice), __LINE__);
	fill_mat<<<dim_grid, dim_block>>>(d_M, d_D, d_S, d_value);

	// Initialise vector mu directly on device
	value = 0.02f;
	check_general_errors(cudaMemcpy(d_value, &value, sizeof(float), cudaMemcpyHostToDevice), __LINE__);
	fill_vec<<<1, D>>>(d_D, d_mu, d_value);

	// Initialise vector sigma directly on device
	value = 0.3f;
	check_general_errors(cudaMemcpy(d_value, &value, sizeof(float), cudaMemcpyHostToDevice), __LINE__);
	fill_vec<<<1, D>>>(d_D, d_sigma, d_value);

	// Initialise vector Y with 0s directly on device (DON'T change this one!)
	value = 0.0f;
	check_general_errors(cudaMemcpy(d_value, &value, sizeof(float), cudaMemcpyHostToDevice), __LINE__);
	int blocks = (M + THREADS - 1)/THREADS;
	fill_vec<<<blocks, THREADS>>>(d_M, d_Y, d_value);

	// Pseudo random number generator for generating random numbers directly on the device
	curandGenerator_t pseudo_rng;
	check_cuRAND_errors(curandCreateGenerator(&pseudo_rng, CURAND_RNG_PSEUDO_DEFAULT), __LINE__);
	// Set the RNG seed
	check_cuRAND_errors(curandSetPseudoRandomGeneratorSeed(pseudo_rng, time(NULL)), __LINE__);

	// Cholesky factorisation to obtain L where L*L.T = P = correlation matrix for the assets
	float *L = perform_cholesky(D);
	check_general_errors(cudaMemcpy(d_L, L, D*D*sizeof(float), cudaMemcpyHostToDevice), __LINE__);

	// Pointer type to an opaque structure holding the cuBLAS library context, must be initialised via
	// cublasCreate prior to running any cuBLAS functions
	cublasHandle_t handle;
	cublasCreate(&handle);

	float mean = 0.0f;
	float std_dev = 1.0f;
	float alpha = 1.0f;
	float beta = 0.0f;
	for (int n = 0; n < N; n++) {
		// Generate and fill Z with ~N(0, 1) randomly generated numbers directly on the device
		check_cuRAND_errors(curandGenerateNormal(pseudo_rng, d_Z, D*M, mean, std_dev), __LINE__);
		check_general_errors(cudaDeviceSynchronize(), __LINE__);

		// Calculate L*Z_n on the device
		// check_cuBLAS_errors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, D, M, D, &alpha, d_L, D, d_Z, D, &beta,
			// d_LZ, D), __LINE__);
		// check_general_errors(cudaDeviceSynchronize(), __LINE__);

		// Calculate (L*Z_n).T = Z_n.T * L.T on the device
		// check_cuBLAS_errors(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, D, M, &alpha, d_LZ, M, &beta, d_LZ, D,
			// d_ZTLT, D), __LINE__);
		// check_general_errors(cudaDeviceSynchronize(), __LINE__);

		// Calculate S_(n+1)^(m, d) = max(S_n^(m, d)*(1 + mu^(d)*dt + sigma^(d)*sqrt(dt)*((L*Z_n).T)^(m, d)), 0)
		// on a single thread on the device
		// euler_timestep<<<dim_grid, dim_block>>>(d_D, d_M, d_dt, d_sqrt_dt,  d_S, d_mu, d_sigma, d_ZTLT);
		euler_timestep<<<dim_grid, dim_block>>>(d_D, d_M, d_dt, d_sqrt_dt,  d_S, d_mu, d_sigma, d_Z);
		check_general_errors(cudaDeviceSynchronize(), __LINE__);
	}

	// Calculate Y^(m) = exp(-r*T)*max(1/D * (S_(N)^(m, 1) + ... + S_(N)^(m, D)) - K, 0) on a single thread on
	// the device
	calc_disc_payoff<<<blocks, THREADS>>>(d_D, d_M, d_K, d_exp__rt, d_S, d_Y);
	check_general_errors(cudaDeviceSynchronize(), __LINE__);

	// Sample mean of Y
	thrust::device_ptr<float> t_d_Y(d_Y);
	const float hat_C_M = thrust::reduce(t_d_Y, t_d_Y + M, 0.0f, thrust::plus<float>())/M;
	// Standard deviation of Y
	check_general_errors(cudaMemcpy(d_hat_C_M, &hat_C_M, sizeof(float), cudaMemcpyHostToDevice), __LINE__);
	calc_sq_dev<<<blocks, THREADS>>>(d_M, d_hat_C_M, d_Y);
	const float hat_sigma_M = sqrt(thrust::reduce(t_d_Y, t_d_Y + M, 0.0f, thrust::plus<float>())/(M - 1));

	// Calculate 95% confidence interval
	// 95th percentile for Normal distribution
	const float z = 1.96;
	const float sqrt_M = sqrt(M);
	// MC + Euler timestepping result
	const float CI_left = hat_C_M - z*hat_sigma_M/sqrt_M;
	const float CI_right = hat_C_M + z*hat_sigma_M/sqrt_M;
	const float radius = z*hat_sigma_M/sqrt_M;

	// Print results
	print_results(hat_C_M, CI_left, CI_right, radius);

	// Free GPU memory
	free_params(d_D, d_M, d_K, d_r, d_dt, d_sqrt_dt, d_exp__rt, d_value, d_hat_C_M);
	free_vecs_mats(d_S, d_mu, d_sigma, d_L, d_Z, d_LZ, d_ZTLT, d_Y, handle);
}

void print_results(const float hat_C_M, const float CI_left, const float CI_right, const float radius) {
	/*
	Neatly prints out MC + Euler timestepping results to the terminal.
	*/
	cout << "MC + Euler result\t\t= " << fixed << setprecision(8) << hat_C_M << flush;
	cout << ",\t\tCI = [" << fixed << setprecision(8) << CI_left << flush;
	cout << ", " << fixed << setprecision(8) << CI_right << flush;
	cout << "],\tRadius = " << fixed << setprecision(8) << radius << endl;
}

void free_params(int* d_D, int *d_M, int *d_K, float *d_r, float *d_dt, float *d_sqrt_dt, float *d_exp__rt,
		float *d_value, float *d_hat_C_M) {
	/*
	Free GPU memory consumed by scalar variables stored on the device.
	*/
	cudaFree(d_D);
	cudaFree(d_M);
	cudaFree(d_K);
	cudaFree(d_r);
	cudaFree(d_dt);
	cudaFree(d_sqrt_dt);
	cudaFree(d_exp__rt);
	cudaFree(d_value);
	cudaFree(d_hat_C_M);
	check_general_errors(cudaDeviceSynchronize(), __LINE__);
}

void free_vecs_mats(float *d_S, float *d_mu, float *d_sigma, float *d_L, float *d_Z, float *d_LZ,
		float *d_ZTLT, float *d_Y, cublasHandle_t handle) {
	/*
	Free GPU memory consumed by vectors and matrices stored on the device.
	*/
	cudaFree(d_S);
	cudaFree(d_mu);
	cudaFree(d_sigma);
	cudaFree(d_L);
	cudaFree(d_Z);
	cudaFree(d_LZ);
	cudaFree(d_ZTLT);
	cudaFree(d_Y);
	cublasDestroy(handle);
	check_general_errors(cudaDeviceSynchronize(), __LINE__);
}
