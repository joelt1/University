clear all;

% Data given in question
x = [5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1];

% Likelihood function for given data
L = @(theta, x, n) (exp(-theta)./(1 - exp(-theta))).^n .* ...
    theta.^(sum(x)) * (1/prod(x));

% Log-likelihood function for given data
log_L = @(theta, x, n) -n*theta + log(theta)*sum(x) - ...
    n*log(1 - exp(-theta)) - sum(log(factorial(x)));

% List of theta to test (very fine partition)
theta = [0:0.0001:6];

% Plot likelihood
subplot(1, 2, 1)
plot(theta, L(theta, x, length(x)))
xlabel("Theta (\theta)")
ylabel("Likelihood (L(\theta))")
title("Likelihood function for the data as a function of theta")
set(gca, 'FontSize', 15)

% Plot log-likelihood
subplot(1, 2, 2)
plot(theta, log_L(theta, x, length(x)))
xlabel("Theta (\theta)")
ylabel("Log-likelihood (log(L(\theta)))")
title("Log-likelihood function for the data as a function of theta")
set(gca, 'FontSize', 15)

% Perform grid search using values for theta and likelihood to obtain the
% MLE of theta
[L_max, idxmax] = max(L(theta, x, length(x)));
theta(idxmax)

% Verify same answer using values for theta and log-likelihood instead
[log_L_max, idxmax] = max(L(theta, x, length(x)));
theta(idxmax)
