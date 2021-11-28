clear all;

% Data from question
x = [0.4453, 9.2865, 0.4077, 2.0623, 10.4737, 5.7525, 2.7159, 0.1954, ...
     0.1608, 8.3143];
n = length(x);

% Posterior distribution of theta (inverse-gamma)
f = @(theta) theta.^(-n) .* exp(-1./theta .* sum(x));

% Plot unnormalised posterior
theta = [0:0.0001:20];
plot(theta, f(theta))
xlabel("Theta (\theta)")
ylabel("Posterior density (f(\theta|x_1, ..., x_n))")
title("Posterior distribution of \theta")
set(gca, 'FontSize', 15)


% b) Want to esimate the 2.5% and 97.5% quantiles of the posterior
% distribution for theta
N = 10^5;
% Burn-in period
B = 1000;

% Use independent sampler because there are finite endpoints to the
% posterior distribution's parameter space i.e. (0, infty).
% Try Inverse-Gamma(n+1, sum(x)) proposals
shape = n+1;
% Have used rate parameterisation of Gamma and Inverse-Gamma distributions
% for a), but MATLAB implements scale parameterisation for inbuilt
% functions --> convert rate to scale using scale = 1/rate
scale = 1/sum(x);
g = @(theta) gampdf(1./theta, shape, scale);


xx = zeros(N, 1);
% Initial point
x = 1;
xx(1) = x;

% Proposal acceptance probability
alpha = @(x, y) min((f(y)*g(x))/(f(x)*g(y)), 1);

% Run independent sampler using MH MCMC algorithm
for t = 2:N
    y = 1/gamrnd(shape, scale);
    % Accept if random uniformly generated number (in [0,1]) is less than
    % acceptance probability
    if rand < alpha(x,y)
        x = y;
    end
    
    xx(t) = x;
end

% Dump first 1000 samples as burn-in
xx = xx(B+1:end);

% Estimated 2.5% and 97.5% quantiles of the posterior distribution for
% theta
lower_quantile = quantile(xx, 0.025)
upper_quantile = quantile(xx, 0.975)
