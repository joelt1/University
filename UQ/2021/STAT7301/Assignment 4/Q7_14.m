clear all;

% Symmetric random walk sampling using q(y|x) = Normal(x, 0.01)
sigma = 0.01;
N = 5000;
% Store random walk path
xx = zeros(N, 1);

% Initial point drawn from N(0, 0.01) distribution
x = randn * sqrt(0.01);
xx(1) = x;

% Target density based on N(10, 2) target distribution
f = @(x) 1/sqrt(2*pi*2) * exp(-1/(2*2) * (x - 10)^2);

% Proposal acceptance probability
alpha = @(x, y) min(f(y)/f(x), 1);

% Run RW sampler
for t = 2:N
    y = randn*sqrt(sigma) + x;
    % Accept if random uniformly generated number (in [0,1]) is less than
    % acceptance probability
    if rand < alpha(x,y)
        x = y;
    end
    
    xx(t) = x;
end

% Plot path taken by random walk
plot([1:N], xx)
xlabel("t")
ylabel("X_t")
title("Path taken by random walk")
set(gca, 'FontSize', 15)

% Burn-in size
B = 1000;
% Generate K estimates of E[log(X^2)]
K = 100;
% Number of confidence intervals to store
M = 20;
% Store M 95% confidence intervals each having a lower and upper bound
CIs = zeros(M, 2);
for i = 1:M
    l_estimates = zeros(K, 1);
    for k = 1:K
        % Store random walk path
        xx = zeros(N, 1);

        % Initial point drawn from N(0, 0.01) distribution
        x = randn * sqrt(0.01);
        xx(1) = x;

        % Run RW sampler
        for t = 2:N
            y = randn*sqrt(sigma) + x;
            % Accept if random uniformly generated number (in [0,1]) is less than
            % acceptance probability
            if rand < alpha(x,y)
                x = y;
            end

            xx(t) = x;
        end

        l_estimates(k) = mean(log(xx(B+1:end).^2));
    end
    
    % Generate i-th confidence interval for the mean
    lower_bound = mean(l_estimates) - 1.96 * (std(l_estimates)/sqrt(K));
    upper_bound = mean(l_estimates) + 1.96 * (std(l_estimates)/sqrt(K));
    CIs(i, :) = [lower_bound, upper_bound];
end

% True value for l
l = 4.58453;
% Find proportion of the time true l is contained within CI
count = 0;
for i = 1:M
    if CIs(i, 1) <= l && l <= CIs(i, 2)
        count = count + 1;
    end
end

contained_proportion = count/M
