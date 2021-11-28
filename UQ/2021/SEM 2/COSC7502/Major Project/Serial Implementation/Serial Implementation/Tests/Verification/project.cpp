/*
Subject - COSC7502
Project Title - HPC in MC Simulation for European Basket Options Pricing
Assessment Title - Serial Implementation (Part A)
Author - Joel Thomas
*/

// Library/module imports
#include <iostream>						// For input-output streaming
#include <ctime>						// Time library, used to set RNG seed
#include <cmath>						// Standard C math library
#include <chrono>						// Manual runtime profiling
#include "../../../Eigen/Core"				// Fast matrix operations
#include "../../../Eigen/Cholesky"			// Perform fast cholesky decomposition
#include "../../../Ziggurat/ziggurat.cpp"		// Fast normal distribution random number generator

// Variable scope
using namespace std;
using namespace Eigen;

// Function headers
void perform_mc_simulation(const int D, const int N, const int M, const ArrayXf S0, const ArrayXf mu, const ArrayXf sigma,
		const int K, const float r, const int T);
void print_results(const float hat_C_M, const float CI_left, const float CI_right, const float radius);

int main(int argc, char *argv[]) {
	/*
	Initialises all simulation variables before passing them onto the main perform_mc_simulation() function
	*/
	char project_topic[] = "HPC in MC Simulation for European Basket Options Pricing - Serial Implementation";
	cout << project_topic << endl;
	cout << "Starting program...\n" << endl;

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

	// Vector of ones
	ArrayXf ones = ArrayXf::Ones(D);

	// Parameters specific to each asset's price dynamics
	const ArrayXf S0 = 100*ones;		// Initial underlying asset prices
	const ArrayXf mu = 0.02*ones;		// Underlying asset drift
	const ArrayXf sigma = 0.3*ones;		// Underlying asset diffusion

	// General parameters
	const int K = 100;					// Contract strike price
	const float r = 0.02;				// Risk-free interest rate
	const int T = 1;					// Contract time to expiry (>= 0)

	// Start profiling
	auto start_time = chrono::high_resolution_clock::now();

	// Run main MC simulation function
	perform_mc_simulation(D, N, M, S0, mu, sigma, K, r, T);

	// Stop profiling
	auto stop_time = chrono::high_resolution_clock::now();
	auto runtime = chrono::duration_cast<chrono::milliseconds>(stop_time - start_time).count();
	cout << "Total runtime\t\t\t= " << runtime << endl;

	return 0;
}

void perform_mc_simulation(const int D, const int N, const int M, const ArrayXf S0, const ArrayXf mu, const ArrayXf sigma,
		const int K, const float r, const int T) {
	/*
	Takes as input the key simulation variables initialised in main() and performs most of the tasks
	appearing in Algorithm 1 in the project report. These include initialising the RNG, performing
	Cholesky decomposition to retrieve L, performing MC + Euler Timestepping, calculating the discounted
	payoff after simulation to time T, generating a 95% confidence interval based on the simulation
	results and finally neatly printing out key results to the console.
	*/
	// Declare variables
	MatrixXf S(M, D);					// Stores M simulated asset prices for N assets
	MatrixXf Z(D, M);					// Stores M correlated standard normal random numbers for N assets
	MatrixXf LZ(M, D);					// Stores L*Z

	// Advanced initisalisation for S, initialise every row of S with S0
	S = MatrixXf::Ones(M, D) * MatrixXf(S0.matrix().asDiagonal());

	// Z_(i, j) ~iid N(mu=0, sigma^2=1) generated via Ziggurat algorithm for RNG
	// These variables are specific to the algorithm and unimportant to the reader
	float fn[128];
	uint32_t kn[128];
	float wn[128];
	r4_nor_setup(kn, fn, wn);
	// Seed for RNG
	uint32_t seed = time(NULL);

	// Find matrix such that L*L^T = P using Cholesky Decomposition
	// Identity matrix is trivial since I*I^T = I*I = I
	// MatrixXf P = MatrixXf::Identity(D, D);
	// MatrixXf L(P.llt().matrixL());
	// Note: correlated standard normal random numbers generated depend on L which in turn depends
	// on the correlation matrix initially provided
	// L = L.matrix();

	// Calculate these constants just once outside loop to avoid expensive recalculation
	const float dt = (float) T/N;		// Stepsize between time intervals
	const ArrayXf mu_dt = mu*dt;
	const ArrayXf sigma_sqrt_dt = sigma*sqrt(dt);
	
	for (int n = 0; n < N; n++) {
		// Advanced initialisation for Z, initialise every element of Z as ~N(0, 1)
		Z = MatrixXf::NullaryExpr(D, M, [&](){ return r4_nor(seed, kn, fn, wn); });
		// LZ = (L * Z).transpose();
		LZ = Z.transpose();

		// Main MC + Euler timestepping loop to simulate single asset
		for (int d = 0; d < D; d++) {
			// Euler timestep asset price
			// S(all, d) = S(all, d).array().cwiseProduct(1 + mu_dt(d) + sigma_sqrt_dt(d)*LZ(all, d).array());
			S(all, d) = S(all, d).array().cwiseProduct(1 + mu_dt(d) + sigma_sqrt_dt(d)*LZ(all, 0).array());
			S(all, d) = S(all, d).cwiseMax(0);
		}
	}

	// Discounted contract payoff for M samples
	ArrayXf Y = exp(-r * T) * (S.rowwise().mean().array() - K).cwiseMax(0);
	// Sample mean and standard deviation of Y
	const float hat_C_M = Y.mean();
	const float hat_sigma_M = sqrt((Y - Y.mean()).square().sum()/(Y.size() - 1));;

	// Calculate 95% confidence interval
	// 95th percentile for Normal distribution
	const float z = 1.96;
	const float sqrt_M = sqrt(M);
	// MC + Euler timestepping result
	const float CI_left = hat_C_M - z*hat_sigma_M/sqrt_M;
	const float CI_right = hat_C_M + z*hat_sigma_M/sqrt_M;
	const float radius = z*hat_sigma_M/sqrt_M;

	// Print results
	cout << "D = " << D << ", N = " << N << ", M = " << M << endl;
	print_results(hat_C_M, CI_left, CI_right, radius);
}

void print_results(const float hat_C_M, const float CI_left, const float CI_right, const float radius) {
	/*
	Neatly prints out MC + Euler timestepping results to the terminal
	*/
	cout << "MC + Euler result\t\t= " << fixed << setprecision(8) << hat_C_M << flush;
	cout << ",\t\tCI = [" << fixed << setprecision(8) << CI_left << flush;
	cout << ", " << fixed << setprecision(8) << CI_right << flush;
	cout << "],\tRadius = " << fixed << setprecision(8) << radius << endl;
}
