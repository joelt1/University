clear all;

% Dataset generated from a mixture of two Gaussians (equal weight)
x = [randn(30, 1); 5 + randn(30, 1)];

% Histogram model (H1)
figure
hist(x, 20)
xlabel("x")
ylabel("Frequency")
title("Histogram of generated dataset")
set(gca, 'FontSize', 18)

% Kernal density estimator 1 (K1)
figure
hold on
[f, y, h] = ksdensity(x);
plot(y, f)
[g, z, h_2] = ksdensity(x, 'Bandwidth', h/2);
plot(z, g)
xlabel("x")
ylabel("Density")
title("Kernel estimators for x")
legend("h=" + h, "h=" + h_2)
set(gca, 'FontSize', 18)
hold off

% GMM - true distribution
p_x = @(x) normpdf(x, 0, 1)*0.5 + normpdf(x, 5, 1)*0.5;

% Kullback-Leibler (KL) divergences between M and K1 and M and K2
div_K1 = sum(p_x(y).*log(p_x(y)./f))
div_K2 = sum(p_x(z).*log(p_x(z)./g))
