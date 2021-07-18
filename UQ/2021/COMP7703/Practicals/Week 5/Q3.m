clear all;

% Generating 2-D datasets as per practical sheet
% a~N([0, 0], [1, 0; 0, 1])
a = randn(200, 2);
% b~N([4, 4], [1, 0; 0, 1])
b = a + 4;
% c~N([-4, -4], [9, 0; 0 ,1])
c = a;
c(:, 1) = 3*c(:, 1);
c = c - 4;
d = [a; b];
e = [a; b; c];

figure
hold on
plot(a(:, 1), a(:, 2), "+")
plot(b(:,1),b(:,2), "o");
plot(c(:,1),c(:,2), "*");
xlabel("x")
ylabel("y")
title("2-D Gaussian Dataset")
hold off

% 1) Testing on dataset d (true means = (0, 0) and (4, 4))
% Randomly initialising T (with points chosen from whole dataset)
% T = [];
% for i = 1:2
%     T = [T; d(randi([1, length(d)]), :)];
% end
% Randomly initialising T (with points chosen from each desired true
% cluster)
T = [d(randi([1, 200]), :); d(randi([201, length(d)]), :)];
lambda = [1, 2, 4];
% Test non-blurring on dataset d with k = 2 clusters and different lambda
for i = 1:length(lambda)
    T = mean_shift(d, T, lambda(i), 0, 2, i + 1)
end

% 2) Testing on dataset e (true means = (0, 0), (4, 4) and (-4, -4))
% Randomly initialising T (with points chosen from each desired true
% cluster)
T = [e(randi([1, 200]), :); e(randi([201, 400]), :); ...
    e(randi([401, length(e)]), :)];
% Test non-blurring on dataset e with k = 3 clusters and different lambda
for i = 1:length(lambda)
    T = mean_shift(e, T, lambda(i), 0, 3, i + 4)
end
