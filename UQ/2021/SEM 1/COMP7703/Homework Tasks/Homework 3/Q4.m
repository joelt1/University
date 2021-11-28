clear all;
train_data = importdata("reg2d.csv");

train_x = train_data(:, 1:2);
train_y = train_data(:, 3);

% Linear fit in 5-D space (z1, z2, z3, z4, z5) corresponds to a quadratic
% fit in 2-D space (x1, x2)
z1 = train_x(:, 1);
z2 = train_x(:, 2);
z3 = z1.*z2;
z4 = z1.^2;
z5 = z2.^2;
Z = [ones(size(z1)), z1, z2, z3, z4, z5];

r = train_y;

% Vector of weights obtained from solving (Z^T * Z)^-1 * Z^T * r
w = (Z.' * Z) \ ((Z.')*r);

% Print SSE
train_error = sum((train_y - Z*w).^2)

% Function for plotting regression model
f = @(Z) Z*w;

% Finely spaced points to use for plotting regression model
z1 = linspace(min(z1), max(z1), 1000).';
z2 = linspace(min(z2), max(z2), 1000).';
z3 = z1.*z2;
z4 = z1.^2;
z5 = z2.^2;
Z = [ones(size(z1)), z1, z2, z3, z4, z5];

% Plot 3-D scatterplot of x1, x2 and r as well 3-D plot of regression
% model
hold on
view(3);
scatter3(train_x(:, 1), train_x(:, 2), train_y)
plot3(z1, z2, f(Z))
xlabel("x1")
ylabel("x2")
zlabel("f(x1, x2)")
title("2-D Quadratic Regression Model on reg2d.csv")
set(gca, 'FontSize', 15)
hold off
