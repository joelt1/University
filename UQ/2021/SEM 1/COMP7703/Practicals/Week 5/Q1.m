clear all;
data = load('heightWeight.mat').heightWeightData;

% Plot the data
figure
hold on
gscatter(data(:, 2), data(:, 3), data(:, 1), "rb")
xlabel("Weight")
ylabel("Height")
title("Height vs. Weight (TRUE)")

% Choose k = 2 clusters and randomly initialise their centres
k1 = [data(randi([1, length(data)]), 2), ...
    data(randi([1, length(data)]), 3)];
k2 = [data(randi([1, length(data)]), 2), ...
    data(randi([1, length(data)]), 3)];

% Plot initial centres with *
scatter(k1(1), k1(2), 400, "k","*")
scatter(k2(1), k2(2), 400, "k", "*")
hold off

prev_ownership = -1;
ownership = zeros(length(data), 1);
num_iter = 0;

f = figure(2);
while ~isequal(prev_ownership, ownership)
    num_iter = num_iter + 1;
    prev_ownership = ownership;
    k1_mean = [];
    k2_mean = [];
    % Calculate ownership of points (whether belong to k1 or k2)
    % Uses standard Euclidean distance formula
    for i = 1:length(data)
        % i
        d1 = sqrt((data(i, 2) - k1(1))^2 + (data(i, 3) - k1(2))^2);
        d2 = sqrt((data(i, 2) - k2(1))^2 + (data(i, 3) - k2(2))^2);
        if d1 <= d2
            ownership(i) = 1;
            k1_mean = [k1_mean; data(i, 2), data(i, 3)];
        elseif d1 > d2
            ownership(i) = 2;
            k2_mean = [k2_mean; data(i, 2), data(i, 3)];
        end
    end
    
    % Plot cluster centres with x
    hold on
    % Uncomment to plot cluster centres and ownership
    gscatter(data(:, 2), data(:, 3), ownership, "gy")
    % Uncomment to plot cluster centres and incorrectly classified points
    % gscatter(data(:, 2), data(:, 3), data(:, 1) == ownership, "rk")
    scatter(k1(1), k1(2), 400, "k", "x")
    scatter(k2(1), k2(2), 400, "k", "x")
    xlabel("Weight")
    ylabel("Height")
    title("Height vs. Weight (PREDICTED)")
    hold off
    pause(0.5)
    clf(f)
    
    k1 = mean(k1_mean);
    k2 = mean(k2_mean);
end

% Classification error (can only calculate this because given targets)
train_error = sum(data(:, 1) == ownership)/length(ownership)

% Plot final cluster centres with x
hold on
% Uncomment to plot cluster centres and ownership
gscatter(data(:, 2), data(:, 3), ownership, "gy")
% Uncomment to plot cluster centres and incorrectly classified points
% gscatter(data(:, 2), data(:, 3), data(:, 1) == ownership, "rk")
scatter(k1(1), k1(2), 400, "k", "x")
scatter(k2(1), k2(2), 400, "k", "x")
xlabel("Weight")
ylabel("Height")
title("Height vs. Weight (PREDICTED)")
hold off
