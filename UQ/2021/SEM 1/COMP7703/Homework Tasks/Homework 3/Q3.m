clear all;
train_data = readtable("BreastCancerTrain.csv");
test_data = readtable("BreastCancerValidation.csv");

% Creating train x and train y
train_x = train_data{:, 1:9};
train_y = train_data{:, 10};

% Creating test x and test y
test_x = test_data{:, 1:9};
test_y = test_data{:, 10};

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
    if train_pred(i) == train_y(i)
        train_acc = train_acc + 1;
    end
end

train_acc = train_acc/length(train_pred);
train_err = 1 - train_acc

% Report testing classification error
test_pred = model.predict(test_x);
test_acc = 0;

for i = 1:length(test_pred)
    if test_pred(i) == test_y(i)
        test_acc = test_acc + 1;
    end
end

test_acc = test_acc/length(test_pred);
test_err = 1 - test_acc
