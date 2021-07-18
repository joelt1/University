clear all;

% Write down y1 as a function of o1 - see assignment
y1 = @(o1) exp(o1)./(exp(o1) + 2*exp(0.2));

% Plot y1 as a function of o1 i.e. y1(o1) overall the interval [-10, 10]
hold on
% Plot function
fplot(y1, [-10, 10])
% Plot x-axis
plot(linspace(-10, 10), linspace(0, 0), "k")
% Plot y-axis
plot(linspace(0, 0), linspace(0, 1), "k")
% Plot midway of function
scatter(log(2) + 0.2, 1/2)
scatter([0:0.25:log(2) + 0.2], 0.5, "k.")
scatter(log(2) + 0.2, [0:0.025:0.5], "k.")
title("Value of the softmax activation function as a function of o_1")
xlabel("o_1")
ylabel("y_1")
set(gca, 'FontSize', 15)
hold off
