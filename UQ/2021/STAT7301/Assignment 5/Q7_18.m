clear all;

% Model parameters
mu_x = 0;
sigma_x = 1;
mu_y = 0;
sigma_y = 1;
% List of rho (correlation values) to test, X and Y independent if p = 0
rho = [0, 0.7, 0.9];
% Joint pdf formula
f = @(x, y, rho) 1/(2*pi*sigma_x*sigma_y*sqrt(1 - rho^2)) * ...
    exp(-1/(2*(1 - rho^2)) .* (((x - mu_x)/sigma_x).^2 - ...
    (2*rho*(x - mu_x).*(y - mu_y))/(sigma_x*sigma_y) + ((y - mu_y)/sigma_y).^2));

% Plot joint pdf
[X, Y] = meshgrid(-5:0.1:5,-5:0.1:5);

figure
surf(X, Y, f(X, Y, rho(1)))
xlabel("X")
ylabel("Y")
zlabel("f_{X, Y}(x, y)")
title("Surface plot of joint PDF of X and Y with \rho = " + rho(1))

% Number of samples to generate
N = 10^4;
% Store generated x and y
xy = [];
% Initial y value (starting point for sampler)
y = 0;

% Run Gibbs sampler
figure
for k = 1:length(rho)
    for i = 1:N
        % (X|Y = y) ~ N(rho*y, 1 - rho^2)
        x = randn*sqrt(1 - rho(k)^2) + rho(k)*y;
        % (Y|X = x) ~ N(rho*x, 1 - rho^2)
        y = randn*sqrt(1 - rho(k)^2) + rho(k)*x;
        xy(i,:) = [x, y];
    end
    
    % Plot generated data
    subplot(1, 3, k)
    hold on
    plot(xy(:,1), xy(:,2), 'bo', 'MarkerSize', 1)
    contour(X, Y, f(X, Y, rho(k)), 'LineColor', 'k', 'LineWidth', 2)
    xlabel("X")
    ylabel("Y")
    title("Plot of data generated using Gibbs Sampler with \rho = " + rho(k))
    set(gca, 'FontSize', 12)
    % Makes axes square and stops stretching of plots (which can make X and
    % Y look dependent even when they are not i.e. rho = 0)
    axis equal
    hold off
end
