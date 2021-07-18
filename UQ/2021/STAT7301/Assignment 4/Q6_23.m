clear all;

% Significance level
alpha = 0.05;
% Critical value
z = norminv(1 - alpha/2);

% Standard interval lower and upper bounds (Problem 5.22 pg. 158)
T1_standard = @(x, n, z) x/n - z*sqrt(((x/n) * (1 - x/n))/n);
T2_standard = @(x, n, z) x/n + z*sqrt(((x/n) * (1 - x/n))/n);

% Score interval lower and upper bounds (Example 6.16 pg. 175)
T1_score = @(x_bar, n, z) (z^2 + 2*n*x_bar - z*sqrt(z^2 - 4*n*(x_bar - 1)*x_bar)) / ...
    (2*(z^2 + n));
T2_score = @(x_bar, n, z) (z^2 + 2*n*x_bar + z*sqrt(z^2 - 4*n*(x_bar - 1)*x_bar)) / ...
    (2*(z^2 + n));


% List of n to test
n = [10, 20, 40, 80];
% Very fine partition for p
p = [0:0.0001:1];

for i = 1:length(n)
    % Create plots for coverage probability for both standard and score
    % intervals given n
    subplot(2, 2, i)
    hold on
    plot(p, cp_standard(p, n(i), z, T1_standard, T2_standard))
    plot(p, cp_score(p, n(i), z, T1_score, T2_score))
    xlabel("p")
    ylabel("P_p(T_1(X) < p < T_2(X))")
    title("Coverage probability for standard and score intervals, " + ...
        "n = " + n(i))
    legend("Standard interval", "Score interval")
    set(gca, 'FontSize', 15)
end

% Function to compute coverage probability for standard interval
function result = cp_standard(p, n, z, T1_standard, T2_standard)
    % Sum from 0 to n, formula taken from Problem 5.22 pg. 158
    result = 0;
    for x = 0:n
        result = result + ...
            (T1_standard(x, n, z) < p & p < T2_standard(x, n, z)) .* ...
            nchoosek(n, x) .* p.^x .* (1 - p).^(n - x);
    end
end

% Function to compute coverage probability for score interval
function result = cp_score(p, n, z, T1_score, T2_score)
    % Sum from 0 to n, formula modified from Problem 5.22 pg. 158
    result = 0;
    for x = 0:n
        % Because dealing with bernoulli data, if x = 1 --> x_bar = 1/n so
        % simply substitute x_bar = x/n for T1_score and T2_score functions
        result = result + ...
            (T1_score(x/n, n, z) < p & p < T2_score(x/n, n, z)) .* ...
            nchoosek(n, x) .* p.^x .* (1 - p).^(n - x);
    end
end
