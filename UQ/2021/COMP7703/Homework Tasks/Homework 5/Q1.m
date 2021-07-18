clear all;

data = importdata("cifar10_data_batch_1.mat");
train_x = double(data.data);
train_y = double(data.labels);

% Student number = 44793203 --> choosing 9, 2, 0, 3 as the classes to
% perform PCA on. Function ismember finds row indices in train_y matching
% these values, then using that to find corresponding rows of train_x.
train_x = train_x(ismember(train_y, [9, 2, 0, 3]), :);
train_y = train_y(ismember(train_y, [9, 2, 0, 3]));

% Running PCA on the dataset
[z, W, lambdas] = pca(train_x);

% a) Plot of the data in the space spanned by the first two principal
% components
figure
gscatter(z(:, 1), z(:, 2), train_y)
xlabel("z_1")
ylabel("z_2")
title("Space spanned by first two principal components (CIFAR-10)")
set(gca, 'FontSize', 15)

% b) Percentage of data variance accounted for by the first two principal
% components
pct_var = sum(lambdas(1:2))/sum(lambdas)

% c) Scree graph
figure
subplot(2, 1, 1)
plot(1:length(W), lambdas, "--xk")
xlabel("Number of eigenvectors")
ylabel("Eigenvalues")
title("Scree graph (CIFAR-10)")
set(gca, 'FontSize', 15)

subplot(2, 1, 2)
plot(1:length(W), cumsum(lambdas)./sum(lambdas), "--xk")
xlabel("Number of eigenvectors")
ylabel("Proportion")
title("Proportion of variance explained (CIFAR-10)")
set(gca, 'FontSize', 15)
