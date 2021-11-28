clear all;
data = readtable("pima_indians_diabetes.csv");

% Student number = 44793203 --> full but shared covariance matrix
% Creating training and testing data
train_data = data(2:500, :);
test_data = data(501:end, :);

% Creating train x and train y
train_x = train_data{:, 1:8};
train_y = train_data{:, 9};

% Creating test x and test y
test_x = test_data{:, 1:8};
test_y = test_data{:, 9};

% Create a linear classifier - full but equal covariances
    % 'linear' = all classes have the same covariance matrix
    % 'diaglinear' = all classes have the same, diagonal covariance matrix
model = fitcdiscr(train_x, train_y, 'DiscrimType', 'linear');
% model = fitcdiscr(train_x, train_y, 'DiscrimType', 'diaglinear');

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
