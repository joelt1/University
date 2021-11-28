clear all;

% Data from question
x = [1.4066, 1.2917, 1.4080, 4.2801, 1.2136, 2.7461, ...
     11.1076, 0.9247, 5.8833, 10.2513, 3.8285, 3.2116, ...
     0.5451, 0.9896, 1.1602, 7.7723, 0.1702, 0.8907, ...
     0.2276, 3.1197, 11.4909, 0.6475, 11.2279, 0.7639];

% b) Estimate T = log(2)/X_tilda, X_tilda = sample median and MLE of lambda
T = log(2)/median(x)
MLE = 1/mean(x)

% c) Perform 10000 bootstrap resamples x* and compute T estimate for and MLE
% of lambda (see b) for each resample
K = 10000;
T_estimates = zeros(K, 1);
ml_estimates = zeros(K, 1);
for k = 1:K
    x_star = datasample(x, length(x));
    T_estimates(k) = log(2)/median(x_star);
    ml_estimates(k) = 1/mean(x_star);
end

% Estimate densities of each estimator using KDE
hold on
kde(T_estimates, 2^4);
kde(ml_estimates, 2^8);
xlabel("Lambda (\lambda)")
ylabel("Density")
title("KDE plots comparing both T and ML estimators")
legend("T estimate", "ML estimate")
set(gca, 'FontSize', 15)
hold off
 