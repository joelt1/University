% European call option pricing under CEV dynamics using a control variate
% with standard GBM dynamics and using Euler time-stepping.
clear all;
format long;

% Model parameters
S0 = 100;
K = 95;
r = 0.05;
sigma = 3;
T = 0.25;

alpha = 0.5;
sigma_star = sigma*S0^(alpha - 1);
M = 5e3;
N = 250;
dt = T/N;

S = S0*ones(M, 1);
% factor = ones(M, 1);
all_Z = zeros(M, N);
% Euler time-stepping
for n = 1:N
    Z = randn(size(S));
    % Store Z to be used for directly calculating S_star
    all_Z(:, n) = Z;
    % Use discretised S
    S = S.*(1 + r*dt + sigma*(S.^(alpha - 1))*sqrt(dt).*Z);
    S = max(S, 0);
    
%     factor = factor.*(1 + r*dt + sigma*sqrt(dt).*Z);
%     factor = max(factor, 0);
end

% Discounted option payoff whose underlying has CEV dynamics
Y = exp(-r*T)*max(K - S, 0);
% S_star = S0*factor;
% Simulating S_star directly using closed form solution for GBM
S_star = S0*exp(N*(r - (sigma_star^2)/2)*dt + ...
    sigma_star*sqrt(dt)*sum(all_Z')');
% Discounted option payoff whose underlying has GBM dynamics
Y_star = exp(-r*T)*max(K - S_star, 0);

% Ordinary MC
hat_C_M = mean(Y); 
hat_sigma_M = std(Y);

[call, put] = my_bls_price(S0, K, T, r, r, sigma_star);
% Explicit known formula used for calculating C_star
C_star = put;

% Optimal coefficient beta_hat
cov_mat = cov(Y, Y_star);
beta_hat = cov_mat(2, 1)/(var(Y_star));
C_cv = Y + beta_hat*(C_star - Y_star);

hat_C_M_cv = mean(C_cv);
hat_sigma_M_cv = std(C_cv);

% 95% Confidence Interval
p = 0.05;
z = norminv(1-p/2);
CI_left = hat_C_M - z * hat_sigma_M/sqrt(M);
CI_right = hat_C_M + z * hat_sigma_M/sqrt(M);
CI_left_cv = hat_C_M_cv - z * hat_sigma_M_cv/sqrt(M);
CI_right_cv = hat_C_M_cv + z * hat_sigma_M_cv/sqrt(M);

% Results
fprintf("MC + Euler results: \n");
fprintf("Price \t\t Std. Error  Radius \t CI \n");
fprintf("%f \t %f \t %f \t [%f, %f] \n\n", hat_C_M, ...
    hat_sigma_M/sqrt(M), z*hat_sigma_M/sqrt(M), CI_left, CI_right);
fprintf("MC + Euler + CV results: \n");
fprintf("Price \t\t Std. Error  Radius \t CI \n");
fprintf("%f \t %f \t %f \t [%f, %f] \n", hat_C_M_cv, ...
    hat_sigma_M_cv/sqrt(M), z*hat_sigma_M_cv/sqrt(M), CI_left_cv, ...
    CI_right_cv);
