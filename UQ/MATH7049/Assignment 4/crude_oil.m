% European call option pricing on a crude oil forward contract using Euler
% time-stepping.
clear all;
format long;

% Model parameters
S0 = 26.9;
K = 23.2;
r = 0.05;
sigma = 0.368;
T1 = 0.5;
T2 = 1;

kappa = 0.472;
alpha = 2.782;
% Initial M - pilot computation
M = 1e3;
N = 100;
dt = T1/N;

X = log(S0)*ones(M, 1);
% Euler time-stepping
for n = 1:N
    Z = randn(size(X));
    % Use discretised X
    X = X + kappa*(alpha - X)*dt + sigma*sqrt(dt)*Z;
end

% Calculate S and F at time T1
S = exp(X);
F = exp(exp(-kappa*(T2 - T1))*log(S) + (1 - exp(-kappa*(T2 - T1))) * ...
    alpha + (sigma^2/(4*kappa)).*(1 - exp(-2*kappa*(T2 - T1))));

% Discounted option payoff
Y = exp(-r*T1)*max(F - K, 0);
hat_sigma_M = std(Y);

% Real computation
M = ceil((hat_sigma_M/0.05)^2);

X = log(S0)*ones(M, 1);
% Euler time-stepping
for n = 1:N
    Z = randn(size(X));
    % Use discretised X
    X = X + kappa*(alpha - X)*dt + sigma*sqrt(dt)*Z;
end

% Calculate S and F at time T1
S = exp(X);
F = exp(exp(-kappa*(T2 - T1))*log(S) + (1 - exp(-kappa*(T2 - T1))) * ...
    alpha + (sigma^2/(4*kappa)).*(1 - exp(-2*kappa*(T2 - T1))));

% Discounted option payoff
Y = exp(-r*T1)*max(F - K, 0);
hat_C_M = mean(Y); 
hat_sigma_M = std(Y);

% 95% Confidence Interval
p = 0.05;
z = norminv(1-p/2);
CI_left = hat_C_M - z * hat_sigma_M/sqrt(M);
CI_right = hat_C_M + z * hat_sigma_M/sqrt(M);

% Results
fprintf("MC + Euler results: \n");
fprintf("Price \t\t Std. Error  Radius \t CI \n");
fprintf("%f \t %f \t %f \t [%f, %f] \n", hat_C_M, hat_sigma_M/sqrt(M), ...
    z*hat_sigma_M/sqrt(M), CI_left, CI_right);
