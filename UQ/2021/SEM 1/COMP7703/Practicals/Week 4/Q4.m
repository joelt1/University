clear all;
data = readtable("iris.txt");

% Fourth column of iris dataset
petal_width = data{:, 4};

% Kernel estimator
figure
hold on
% Default width = 0.4035
[f, x, h] = ksdensity(petal_width);
plot(x, f)
% Larger than default width
[f, x, h] = ksdensity(petal_width, 'Bandwidth', 0.5);
plot(x, f)
% Smaller than default width
[f, x, h] = ksdensity(petal_width, 'Bandwidth', 0.1);
plot(x, f)
% Smaller than default width
[f, x, h] = ksdensity(petal_width, 'Bandwidth', 0.3);
plot(x, f)
xlabel("Petal width")
ylabel("Density")
title("Kernel estimators for petal width")
legend("h=0.4035", "h=0.5", "h=0.1", "h=0.3")
set(gca, 'FontSize', 18)
hold off

% Basically larger h = more smooth, smaller h = captures peaks better
