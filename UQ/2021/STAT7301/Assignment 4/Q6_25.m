clear all;

% Data given in question
x = [125, 18, 20, 34];

% Likelihood function for given data
L = @(theta, x) ((2 + theta)/4).^x(1) .* ((1 - theta)/4).^x(2) .* ...
    ((1 - theta)/4).^x(3) .* (theta/4).^x(4);

% Log-likelihood function for given data
log_L = @(theta, x) x(1)*log(2 + theta) + ...
    (x(2) + x(3))*log(1 - theta) + x(4)*log(theta);

% Score statistic
score = @(theta, x) x(4)./theta + x(1)./(theta + 2) - ...
    (x(2) + x(3))./(1 - theta);

% Hessian function
hessian = @(theta, x) -x(4)./theta.^2 - x(1)./(theta + 2).^2 - ...
    (x(2) + x(3))./(1 - theta).^2;

% List of theta (very fine partition)
thetas = [0.4:0.0001:0.8];

% Plot likelihood
subplot(2, 2, 1)
plot(thetas, L(thetas, x))
xlabel("Theta (\theta)")
ylabel("Likelihood (L(\theta))")
title("Likelihood function for the data as a function of theta")
set(gca, 'FontSize', 15)

% Plot log-likelihood
subplot(2, 2, 2)
plot(thetas, log_L(thetas, x))
xlabel("Theta (\theta)")
ylabel("Log-likelihood (log(L(\theta)))")
title("Log-likelihood function for the data as a function of theta")
set(gca, 'FontSize', 15)

% Plot score
subplot(2, 2, 3)
plot(thetas, score(thetas, x))
xlabel("Theta (\theta)")
ylabel("Score (S(\theta)))")
title("Score statistic for the data as a function of theta")
set(gca, 'FontSize', 15)

% Plot hessian
subplot(2, 2, 4)
plot(thetas, hessian(thetas, x))
xlabel("Theta (\theta)")
ylabel("Hessian (H(\theta)))")
title("Hessian function for the data as a function of theta")
set(gca, 'FontSize', 15)

% b) Perform Newton-Raphson procedure to find the MLE of theta
% Need f and f' for this procedure --> use score and hessian functions.
% Don't use log-likehood and score because log-likehood missing constant
% term.
theta = 0.1;
new_theta = 0.2;
while theta ~= new_theta
    theta = new_theta;
    new_theta = theta - score(theta, x)/hessian(theta, x);
end

theta

% c) Perform grid search using values for theta and likelihood to obtain
% the MLE of theta
[L_max, idxmax] = max(log_L(thetas, x));
theta = thetas(idxmax)
