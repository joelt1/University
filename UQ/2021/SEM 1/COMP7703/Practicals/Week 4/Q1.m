% IMPORTANT NOTES REGARDING DISCRIMINANT ANALYSIS
% The model for discriminant analysis is:
    % Each class (Y) generates data (X) using a multivariate normal
    % distribution. That is, the model assumes X has a Gaussian mixture
    % distribution (gmdistribution).

    % For linear discriminant analysis, the model has the same covariance
    % matrix for each class, only the means vary.

    % For quadratic discriminant analysis, both means and covariances of
    % each class vary.
    
% https://au.mathworks.com/help/stats/fitcdiscr.html
clear all;
data = readtable("pima_indians_diabetes.csv");

% Creating training and testing data
train_data = data(2:500, :);
test_data = data(501:end, :);

% Creating train x and train y
train_x = train_data{:, 1:8};
train_y = train_data{:, 9};

% Creating test x and test y
test_x = test_data{:, 1:8};
test_y = test_data{:, 9};

% Create a quadratic classifier
    % 'quadratic' = covariance matrices can vary among classes
    % 'diagquadratic' = covariance matrices are diagonal and can vary among
        % classes
model = fitcdiscr(train_x, train_y, 'DiscrimType', 'quadratic');
% model = fitcdiscr(train_x, train_y, 'DiscrimType', 'diagquadratic');

% Verify targets
model.ClassNames

% Report training classification error
train_pred = model.predict(train_x);
train_acc = 0;

for i = 1:length(train_pred)
    if strcmp(train_pred(i), train_y(i))
        train_acc = train_acc + 1;
    end
end

train_acc = train_acc/length(train_pred);
train_err = 1 - train_acc

% Report testing classification error
test_pred = model.predict(test_x);
test_acc = 0;

for i = 1:length(test_pred)
    if strcmp(test_pred(i), test_y(i))
        test_acc = test_acc + 1;
    end
end

test_acc = test_acc/length(test_pred);
test_err = 1 - test_acc

% EXTRA CODE:

% Plot scatter plot between two columns colour grouped by target
% figure
% hold on
% plot_1 = gscatter(train_x(:, 1), train_x(:, 2), train_y, 'kr', 'o^', ...
%     [], 'on');
% xlabel("x1")
% ylabel("x2")
% title("Features x2 vs. x1 grouped by target")

% K = model.Coeffs(1, 2).Const;
% L = model.Coeffs(1, 2).Linear;
% M = model.Coeffs(1, 2).Quadratic;

% Plot the curve that separates the first and second classes
% f = @(x) (x.')*M*x + (L.')*x + K;
% f = @(x1, x2) K + L(1)*x1 + L(2)*x2;

% f_x = zeros(length(train_x), 1);
% 
% for i = 1:length(train_x)
%     f_x(i) = f(train_x(i, 1:2).');
% end
