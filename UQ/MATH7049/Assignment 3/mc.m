% Prices the same asset-or-nothing call option from Q2 using an ordinary
% Monte Carlo (MC) method.
clear all;
format  long;

S0 = 100;                       % Initial asset price
K = 100;                        % Strike price
r = 0.02;                       % Interest rate
sigma = 0.2;                    % Option volatility
T = 2;                          % Time to expiry
M_list = [2000 .*2.^[0:6]];     % Number of samples

% 95% confidence interval - see L5.11
p = 0.05;
z = norminv(1 - p/2);
for m = 1:length(M_list)
    % See L5.15 for relevant formula
    Y = S0 .* exp((r - sigma^2/2).* T + sigma * sqrt(T).* ...
        randn(M_list(m), 1));
    % Asset-or-nothing call option payoff at maturity
    for y = 1:length(Y)
        if Y(y) < K
            Y(y) = 0;
        end
    end
    % Discounted payoff at time 0
    Y = exp(-r * T) .* Y;

    % Sample mean
    hat_C_M(m) = mean(Y);
    % Sample std
    hat_sigma_M(m) = std(Y);

    % Confidence interval - left and right boundaries
    CI_left(m)  = hat_C_M(m) - z * hat_sigma_M(m)/sqrt(M_list(m));
    CI_right(m) = hat_C_M(m) + z * hat_sigma_M(m)/sqrt(M_list(m));
    radius(m) = z * hat_sigma_M(m)/sqrt(M_list(m));
end

% Exact option price
exact_call = S0 * normcdf((log(S0/K) + (r + (sigma^2)/2)*T)/(sigma*T^(1/2)));
disp(sprintf("Exact asset-or-nothing call price: %.9g \n", exact_call));

% Output
fprintf("M \t\t  Value \t  Radius \t Ratios \n");
for m = 1:length(M_list)
    if M_list(m) < 10000
        fprintf("%1d \t %3.6f \t %3.6f", M_list(m), hat_C_M(m), radius(m));
    elseif M_list(m) < 100000
        fprintf("%1d \t %3.6f \t %3.6f", M_list(m), hat_C_M(m), radius(m));
    else
        fprintf("%1d \t %3.6f \t %3.6f", M_list(m), hat_C_M(m), radius(m));
    end
       
    if (m > 1)
        ratio(m-1) = radius(m-1)/radius(m);
        fprintf("\t %3.3f", ratio(m-1));
    end
    
    fprintf("\n");
end

% Plot results
plot(M_list, hat_C_M);
xlabel("M");
ylabel("$$\hat{C}_M$$", "Interpreter", "Latex");
