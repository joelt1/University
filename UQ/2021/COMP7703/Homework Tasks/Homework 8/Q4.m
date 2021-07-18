clear all;

% Read in dataset
data = readtable("hw8.csv");
% Angle between x (first 500 columns) and x_dash (next 500 columns) stored
% in final column (1001th column)
thetas = data{1:end, end};

% Define function to calculate k_ac
k_ac = @(x, x_dash, theta) (norm(x)*norm(x_dash))/(2*pi) * ...
    (sin(theta) + (pi - theta)*cos(theta));

% Store 500 values in list (500 samples total in data)
k_acs = [];
% Calculate k_ac first
for j = 1:height(data)
    x = data{j, 1:500}';
    x_dash = data{j, 501:1000}';
    theta = data{j, end};
    
    k_acs = [k_acs; k_ac(x, x_dash, theta)];
end

% List of n to test for k_approx
n = [10, 100, 1000, 10000];
for i = 1:length(n)
    % Store 500 values in list (500 samples total in data)
    k_approxs = [];
    
    % Calculate k_approx for given n
    for j = 1:height(data)
        x = data{j, 1:500}';
        x_dash = data{j, 501:1000}';
        theta = data{j, end};

        % See k_approx.m function
        k_approxs = [k_approxs; k_approx(x, x_dash, n(i))];
    end

    % Create a subplot comparing k_ac vs. k_approx for given n
    subplot(2, 2, i)
    hold on
    plot(thetas, k_acs, "LineWidth", 3)
    scatter(thetas, k_approxs, 20)
    xlabel("Theta (\theta)")
    ylabel("Kernel function")
    title("k_{ac} vs. k_{approx} for n = " + n(i))
    legend("k_{ac}", "k_{approx}")
    set(gca, 'FontSize', 15)
    hold off
end
