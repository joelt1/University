clear all;

% Model parameters
A = 1;
nu = 1;

% Number of samples to generate
N = 100000;

% Store samples and running mean for sigma^2
sigma = [];
sigma_bar = [];

% Run sampler for hierarchical model
for i = 1:N
    % Have used rate parameterisation of Inverse-Gamma distribution but
    % MATLAB implements scale parameterisation for inbuilt functions
    % --> convert rate to scale using scale = 1/rate
    a = 1/gamrnd(1/2, A^2);
    sigma(i) = sqrt(1/gamrnd(nu/2, a/nu));
    %
    sigma_bar(i) = mean(sigma(1:i));
end

% Plot KDE for generated samples
figure
kde(sigma, 2^2)
xlabel("\sigma")
ylabel("Density")
title("KDE plot for 100000 generated samples of \sigma")
set(gca, 'FontSize', 15)

% Plot running sample mean
figure
plot([1:100000], sigma_bar)
xlabel("n")
ylabel('$\bar{\sigma}_{(n)}$','Interpreter','Latex')
title("Running sample mean for \sigma")
set(gca, 'FontSize', 15)

% Converged value for running sample mean for sigma^2
converged_sigma_mean = sigma_bar(end)
