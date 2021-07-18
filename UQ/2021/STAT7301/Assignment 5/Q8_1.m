clear all;

% List of alpha and beta to test
alpha = [0.5, 1.5];
beta = [0.5, 1.5];
% Fine partition to plot over
x = [0:0.0001:1];

% Test each alpha and beta combination for the Beta distribution
for i = 1:2
    subplot(1, 2, i)
    plot(x, betapdf(x, alpha(i), beta(i)))
    xlabel("X")
    ylabel("f_X(x)")
    title("Beta distribution for \alpha = " + alpha(i) + ", \beta = " + ...
        beta(i))
    set(gca, 'FontSize', 16)
end
