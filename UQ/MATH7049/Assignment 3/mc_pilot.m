% Prices the same asset-or-nothing call option from Q2 using an ordinary
% Monte Carlo (MC) method. First estimates how large M should be so that
% estimated price of the option will be with +- $0.1 at a confidence
% interval of 95%. Then carries out the computation with this M.
clear all;
format  long;

S0 = 100;           % Initial asset price
K = 100;            % Strike price
r = 0.02;           % Interest rate
sigma = 0.2;        % Option volatility
T = 2;              % Time to expiry
M = 2000 * 2^6;     % Initial number of samples

% Pilot computation - since at M = 2000 * 2^6 = 128000, we know that the
% radius > 0.1 (from Q3.a results), we can use this as a safe initial value
% to estimate hat_sigma_M which is required for estimating M itself.
Y = S0 .* exp((r - sigma^2/2).* T + sigma * sqrt(T).* randn(M, 1));
% Asset-or-nothing call option payoff at maturity
for y = 1:length(Y)
    if Y(y) < K
        Y(y) = 0;
    end
end
% Discounted payoff at time 0
Y = exp(-r * T) .* Y;

% Sample mean
hat_C_M = mean(Y);
% Sample std
hat_sigma_M = std(Y);

% 95% confidence interval - see L5.11
p = 0.05;
z = norminv(1 - p/2);
% Left and right boundaries
CI_left  = hat_C_M - z * hat_sigma_M/sqrt(M);
CI_right = hat_C_M + z * hat_sigma_M/sqrt(M);

% Estimate large enough M at desired radius of 0.1 - formula used is
% rearranged formula for calculating radius. Ceiling function applied to
% get next highest integer.
M = ceil((z * hat_sigma_M / 0.1)^2);
disp(hat_sigma_M)

% Repeat computation using new M
Y = S0 .* exp((r - sigma^2/2).* T + sigma * sqrt(T).* randn(M, 1));
for y = 1:length(Y)
    if Y(y) < K
        Y(y) = 0;
    end
end
Y = exp(-r * T) .* Y;

% Sample mean
hat_C_M = mean(Y);
% Sample std
hat_sigma_M = std(Y);

% 95% confidence interval - see L5.11
p = 0.05;
z = norminv(1 - p/2);
% Left and right boundaries
CI_left  = hat_C_M - z * hat_sigma_M/sqrt(M);
CI_right = hat_C_M + z * hat_sigma_M/sqrt(M);
% Verify correct radius - will be slightly above or less than 0.1 because
% re-estimating hat_sigma_M using new M.
radius = z * hat_sigma_M/sqrt(M);

% Exact option price
exact_call = S0 * normcdf((log(S0/K) + (r + (sigma^2)/2)*T)/(sigma*T^(1/2)));
disp(sprintf("Exact asset-or-nothing call price: %.9g \n", exact_call));

% Output
fprintf("M \t\t\t  Value \t\t\t  CI \t\t\t\t Radius \n");
fprintf("%1d \t %3.6f \t [%3.6f, %3.6f] \t %3.6f", M, hat_C_M, CI_left, ...
    CI_right, radius);
fprintf("\n");
