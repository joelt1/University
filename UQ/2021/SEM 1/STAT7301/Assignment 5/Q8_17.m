clear all;

% Data
x = [-0.4326, -1.6656, 0.1253, 0.2877, -1.1465];

% Model parameters
n = length(x);
% Prior of mu ~ N(0, sigma0^2)
sigma0 = 1;
% Prior of sigma^2 ~ InvGamma(alpha0, lambda0)
alpha0 = 1;
lambda0 = 1;

% Number of samples to generate
N = 10^3;

% Evenly spaced partition for y
y = linspace(-5, 5, 1000);
% Store estimated likelihoods for Y (parameters generated via Gibbs sampler
% below)
f_y_given_theta = [];

% Initial sigma value for Gibbs sampler
sigma = 1;
% Run Gibbs sampler
for i = 1:N
    % (mu|sigma^2, x) ~ Normal, see equation 8.9 on page 235
    kappa = sigma^2/(sigma0^2/n);
    mu_mean = mean(x)/(1 + kappa);
    mu_sigma2 = (sigma^2/n)/(1 + kappa);
    % Use square root to get standard deviation needed as parameter for
    % normrnd function
    mu = normrnd(mu_mean, sqrt(mu_sigma2));
    
    % (sigma^2|mu, x) ~ InvGamma, see equation 8.10 on page 235
    sigma2_shape = alpha0 + n/2;
    sigma2_rate = sum((x - mu).^2)/2 + lambda0;
    % Have used rate parameterisation of Inverse-Gamma distribution but
    % MATLAB implements scale parameterisation for inbuilt functions
    % --> convert rate to scale using scale = 1/rate
    sigma2_scale = 1/sigma2_rate;
    % sigma2 ~ InvGamma --> 1/sigma2 ~ Gamma
    sigma2 = 1/gamrnd(sigma2_shape, sigma2_scale);
    
    % Y ~ N(mu, sigma^2), store resulting pdf
    f_y_given_theta(i, :) = normpdf(y, mu, sqrt(sigma2));    
end

% Monte Carlo approximation for f(y|x)
f_y_given_x = mean(f_y_given_theta);

% Plot and compare results - common sense pdf is unreliable since VERY
% SMALL sample size (n = 5) used to estimate mean and variance
hold on
plot(y, f_y_given_x)
plot(y, normpdf(y, mean(x), std(x)))
xlabel("Y")
ylabel("f(y|x)")
title("Comparison of estimated predictive pdf vs. 'common-sense' " + ...
    "Gaussian pdf")
legend("Predictive pdf", "Common-sense Gaussian pdf")
set(gca, 'FontSize', 12)
hold off
