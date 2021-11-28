clear all;
data = readtable("iris.txt");

% Fourth column of iris dataset
petal_width = data{:, 4};

% Testing different bin sizes for histograms
figure
hist(petal_width)
% hist(petal_width, 2)
% hist(petal_width, 4)
% hist(petal_width, 8)
% hist(petal_width, 10) % default
% hist(petal_width, 20)
% hist(petal_width, 50)
% hist(petal_width, 100)
% hist(petal_width, 200)
xlabel("Petal width")
ylabel("Frequency")
title("Histogram for petal width")
set(gca, 'FontSize', 18)

% Kernel estimator
figure
hold on
% Default width = 0.4035
[f, x, h] = ksdensity(petal_width);
plot(x, f)
xlabel("Petal width")
ylabel("Density")
title("Kernel estimator for petal width")
legend("h=0.4035")
set(gca, 'FontSize', 18)
