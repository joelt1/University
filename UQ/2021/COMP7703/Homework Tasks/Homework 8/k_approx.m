function [result] = k_approx(x, x_dash, n)
    % k_approx function to use in Q4.m
    % Inputs:
        % x = first data sample (vector)
        % x_dash = second data sample (vector)
        % n = number of w_i draws from N(0, 1) distribution
    % Output
        % result = value of k_approx
    
    result = 0;
    % Following formula given to calculate k_approx (see Homework 8 Q4)
    for i = 1:n
        % size(w_i) = size(x)' = (1, 500), each element in w_i ~ N(0, 1)
        w_i = randn(size(x))';
        result = result + max(0, dot(w_i, x)) * max(0, dot(w_i, x_dash));
    end
    
    % Find average from sum (see formula in Homework 8 Q4)
    result = result/n;        
end
