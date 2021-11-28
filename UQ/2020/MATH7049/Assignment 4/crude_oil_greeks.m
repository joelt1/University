% European call option pricing on a crude oil forward contract using Euler
% time-stepping - also calculates delta and gamma.
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
dS = 0.001;

X = log(S0)*ones(M, 1);
% Euler time-stepping
for n = 1:N
    Z = randn(size(X));
    % Use discretised X
    X = X + kappa*(alpha - X)*dt + sigma*sqrt(dt)*Z;
end

% Calculate S and F at time T1
S = exp(X);
F = exp(exp(-kappa*(T2 - T1))*log(S) + ...
    (1 - exp(-kappa*(T2 - T1))) * alpha + ...
    (sigma^2/(4*kappa))*(1 - exp(-2*kappa*(T2 - T1))));

% Discounted option payoff
Y = exp(-r*T1)*max(F - K, 0);
hat_sigma_M = std(Y);

% Real computation
M = ceil((hat_sigma_M/0.05)^2);

X = log(S0)*ones(M, 1);
% Simulation starting from S0 + dS
X_plus = log(S0 + dS)*ones(M, 1);
% Simulation starting from S0 - dS
X_minus = log(S0 - dS)*ones(M, 1);
% Euler time-stepping
for n = 1:N
    Z = randn(size(X));
    % Use discretised X
    X = X + kappa*(alpha - X)*dt + sigma*sqrt(dt)*Z;
    X_plus = X_plus + kappa*(alpha - X_plus)*dt + sigma*sqrt(dt)*Z;
    X_minus = X_minus + kappa*(alpha - X_minus)*dt + sigma*sqrt(dt)*Z;
end

% Calculate S and F at time T1 - S0 start
S = exp(X);
F = exp(exp(-kappa*(T2 - T1))*log(S) + ...
    (1 - exp(-kappa*(T2 - T1))) * alpha + ...
    (sigma^2/(4*kappa))*(1 - exp(-2*kappa*(T2 - T1))));

% Calculate S and F at time T1 - S0 + dS start
S_plus = exp(X_plus);
F_plus = exp(exp(-kappa*(T2 - T1))*log(S_plus) + ...
    (1 - exp(-kappa*(T2 - T1))) * alpha + ...
    (sigma^2/(4*kappa))*(1 - exp(-2*kappa*(T2 - T1))));

% Calculate S and F at time T1 - S0 - dS start
S_minus = exp(X_minus);
F_minus = exp(exp(-kappa*(T2 - T1))*log(S_minus) + ...
    (1 - exp(-kappa*(T2 - T1))) * alpha + ...
    (sigma^2/(4*kappa))*(1 - exp(-2*kappa*(T2 - T1))));

% Discounted option payoff
Y = exp(-r*T1)*max(F - K, 0);
Y_plus = exp(-r*T1)*max(F_plus - K, 0);
Y_minus = exp(-r*T1)*max(F_minus - K, 0);

hat_C_M = mean(Y); 
hat_sigma_M = std(Y);

hat_delta = (Y_plus - Y)/dS;
hat_delta_M = mean(hat_delta);
hat_delta_sigma_M = std(hat_delta);

hat_gamma = (Y_plus - 2*Y + Y_minus)/dS^2;
hat_gamma_M = mean(hat_gamma);
hat_gamma_sigma_M = std(hat_gamma);

% 95% Confidence Intervals
p = 0.05;
z = norminv(1-p/2);
% For option price
CI_left = hat_C_M - z * hat_sigma_M/sqrt(M);
CI_right = hat_C_M + z * hat_sigma_M/sqrt(M);
% For option delta
CI_left_delta = hat_delta_M - z * hat_delta_sigma_M/sqrt(M);
CI_right_delta = hat_delta_M + z * hat_delta_sigma_M/sqrt(M);
% For option gamma
CI_left_gamma = hat_gamma_M - z * hat_gamma_sigma_M/sqrt(M);
CI_right_gamma = hat_gamma_M + z * hat_gamma_sigma_M/sqrt(M);

% Results
fprintf("MC + Euler results: \n");
fprintf("Price \t\t Std. Error  Radius \t CI \n");
fprintf("%f \t %f \t %f \t [%f, %f] \n\n", hat_C_M, hat_sigma_M/sqrt(M), ...
    z*hat_sigma_M/sqrt(M), CI_left, CI_right);
fprintf("Delta \t\t Std. Error  Radius \t CI \n");
fprintf("%f \t %f \t %f \t [%f, %f] \n\n", hat_delta_M, ...
    hat_delta_sigma_M/sqrt(M), z*hat_delta_sigma_M/sqrt(M), ...
    CI_left_delta, CI_right_delta);
fprintf("Gamma \t\t Std. Error  Radius \t CI \n");
fprintf("%f \t %f \t %f \t [%f, %f] \n", hat_gamma_M, ...
    hat_gamma_sigma_M/sqrt(M), z*hat_gamma_sigma_M/sqrt(M), ...
    CI_left_gamma, CI_right_gamma);
