% European call option pricing using an importance sampling Monte Carlo
% algorithm.
clear all;
format long;

% Model parameters
S0 = 100;
K = 150;
r = 0.01;
sigma = 0.1;
T = 1;
M = 1e5;

% Calculate lambda from C0 and C0_tilda
[call, put] = my_bls_price(S0, K, T, r, r, sigma);
C0 = call;
[call, put] = my_bls_price(S0, K, T, r + sigma^2, 0, sigma);
C0_tilda = S0*call;
lambda = (log(C0_tilda/(S0*C0)) - r*T)*(sigma*T);

% Ordinary MC using importance sampling
Z = randn(M, 1);
S = S0*exp((r + sigma*lambda - sigma^2/2)*T + sigma*sqrt(T)*Z);
% Discounted option payoff
Y = exp(-r*T)*max(S - K, 0);

hat_C_M = mean(Y); 
hat_sigma_M = std(Y);

% 95% Confidence Interval
p = 0.05;
z = norminv(1-p/2);
CI_left = hat_C_M - z * hat_sigma_M/sqrt(M);
CI_right = hat_C_M + z * hat_sigma_M/sqrt(M);

% Results
fprintf("MC + IS results: \n");
fprintf("Price \t\t Std. Error  Radius \t CI \n");
fprintf("%f \t %f \t %f \t [%f, %f] \n", hat_C_M, hat_sigma_M/sqrt(M), ...
    z*hat_sigma_M/sqrt(M), CI_left, CI_right);
