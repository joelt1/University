clear all;

% Target density for random walk sampler
f = @(x1, x2) exp(-(x1.^2.*x2.^2 + x1.^2 + x2.^2 - 8*x1 - 8*x2)/2);

% a) 3D and contour plots for given two-dimensional pdf
% 3D plot
x1 = [0:0.0001:10];
x2 = [0:0.0001:10];

figure
subplot(1, 2, 1)
plot3(x1, x2, f(x1, x2))
xlabel("x_1")
ylabel("x_2")
zlabel("f(x_1, x_2)")
title("3D plot for two-dimensional pdf")
set(gca, 'FontSize', 15)

% Contour plot
[x1, x2] = meshgrid(linspace(0, 10), linspace(0, 10));

subplot(1, 2, 2)
contour(x1, x2, f(x1, x2))
xlabel("x_1")
ylabel("x_2")
title("Contour plot for two-dimensional pdf")
set(gca, 'FontSize', 15)

% b) Implement a 2D random walk sampler
sigma = 0.2;
N = 5000;
% Store 2D random walk path
xx = zeros(N, 2);

% Initial point
x = [0, 4];
xx(1, :) = x;

% Proposal acceptance probability (x, y are positional vectors of length 2)
alpha = @(x, y) min(f(y(1), y(2))/f(x(1), x(2)), 1);

% Run 2D RW sampler
for t = 2:N
    y = x + sigma*mvnrnd(zeros(2, 1), eye(2), 1);
    % Accept if random uniformly generated number (in [0,1]) is less than
    % acceptance probability
    if rand < alpha(x,y)
        x = y;
    end
    
    xx(t, :) = x;
end

% Plot path taken by 2D random walk
% figure
% plot3([1:N], xx(:, 1), xx(:, 2))
% xlabel("t")
% ylabel("X_1")
% zlabel("X_2")
% title("Path taken by 2D random walk")
% set(gca, 'FontSize', 15)

% c) Plot progression of the first component of the MC against time.
% Progression using sigma = 0.2 already stored in xx, store for sigma = 2
% in yy
sigma = 2;
yy = zeros(N, 2);

% Initial point
x = [0, 4];
yy(1, :) = x;

% Run 2D RW sampler
for t = 2:N
    y = x + sigma*mvnrnd(zeros(2, 1), eye(2), 1);
    % Accept if random uniformly generated number (in [0,1]) is less than
    % acceptance probability
    if rand < alpha(x,y)
        x = y;
    end
    
    yy(t, :) = x;
end

figure
subplot(2, 1, 1)
plot([1:N], xx(:, 1))
xlabel("t")
ylabel("X_1")
title("Path taken by first component of Markov Chain against time, " + ...
    "\sigma = 0.2")
set(gca, 'FontSize', 15)

subplot(2, 1, 2)
plot([1:N], yy(:, 1))
xlabel("t")
ylabel("X_1")
title("Path taken by first component of Markov Chain against time, " + ...
    "\sigma = 2")
set(gca, 'FontSize', 15)

% d) KDE of the pdf X_1
% Burn-in period
B = 1000;
figure
kde(xx(B+1:end, 1), 2^6);
xlabel("X_1")
ylabel("Density")
title("KDE plot of the pdf X_1")
set(gca, 'FontSize', 15)
