clear all;

data = readtable("iris.txt");
train_x = data{:, 1:2};
train_y = data{:, 5};

% Each m_i is d-dim --> d x 1
% Mean of class 1 = Iris-setosa
m1 = mean(train_x(1:50, :))';
% Mean of class 2 = Iris-versicolor
m2 = mean(train_x(50:100, :))';
% Mean of class 3 = Iris-virginica
m3 = mean(train_x(101:end, :))';

% Store in array mi representing m_i from textbook, i = 1:3
mi = [m1, m2, m3];

% Scatter of the means, taking sum along horizontal instead of vertical
% axis
m = sum(mi, 2)/3;

% N1 = N2 = N3 = N = 50 (recall r^t_i = 1 if x^t is an element of C_i, 0
% otherwise and there are 50 samples of each class in Iris dataset)
N = 50;

SB = zeros(2, 2);
% Between-class scatter matrix BEFORE projection
for i = 1:3
    SB = SB + N * (mi(:, i) - m) * (mi(:, i) - m)';
end

% Display S_B
SB
