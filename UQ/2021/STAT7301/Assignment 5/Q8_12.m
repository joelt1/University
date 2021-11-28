clear all;

% c) Generate an iid sample of size 100 for the zero-inflated Poisson
% model.
% Given parameters, note that these are the true values for p and lambda
% (for our experiment)
p = 0.3;
lambda = 2;
n = 100;

% Generate iid sample
r = [];
x = [];
for i = 1:n
    % x_i = r_i * y_i for zero-inflated Poisson model
    % r_i ~ Ber(p), y_i ~ Poi(lambda)
    r(i) = binornd(1, p);
    x(i) = r(i) * poissrnd(lambda);
end

% d) Implement Gibbs sampler to generate a large dependent sample from the
% posterior and use this to construct 95% credible intervals for p and
% lambda using generated data from c).
% Model parameters
a = 1;
b = 1;

% Number of p, lambda samples to generate
N = 10000;

% Store generated p and lambda
ps = [];
lambdas = [];

% Run Gibbs Sampler
for i = 1:N
    % Have used rate parameterisation of Gamma distribution but MATLAB
    % implements scale parameterisation for inbuilt functions --> convert
    % rate to scale using scale = 1/rate
    % Sample lambda using conditional distribution
    lambdas(i) = gamrnd(a + sum(x), 1/(b + sum(r)));
    % Sample p using conditional distribution
    ps(i) = betarnd(1 + sum(r), 1 + n - sum(r));
    
    % Sample r (vector) using conditional distribution
    for k = 1:n
        r(k) = binornd(1, ...
            (ps(i)*exp(-lambdas(i)))/(ps(i)*exp(-lambdas(i)) + (1 - ps(i))*(x(k) == 0)));
    end
end

% Construct 95% credible interval for p
p_lower = quantile(ps, 0.025)
p_upper = quantile(ps, 0.975)

% Construct 95% credible interval for lambda
lambda_lower = quantile(lambdas, 0.025)
lambda_upper = quantile(lambdas, 0.975)
