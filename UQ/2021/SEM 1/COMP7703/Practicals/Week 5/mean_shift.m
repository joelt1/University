function [T] = mean_shift(S, T, lambda, blurring, k, fig_id)
    % Implementation of the mean shift clustering algorithm as discussed in
    % the lectures. Uses a flat/gaussian kernel function with pre-specified
    % value for the radius parameter lambda. When T = S -> blurring
    % processing otherwise non-blurring process.
    % Inputs:
        % S = dataset/sample
        % T = randomised cluster centres or = S
        % lambda = radius parameter
        % blurring = flag to indicate whether blurring or non-blurring process
        % k = number of desired classes of non-blurring process
        % fig_id = used to create new figure windows for each test run
    % Outputs:
        % T = cluster centres
    
    % Defining flat kernel (uncomment to use this)
    K = @(x) norm(x) <= lambda;
    % Defining Gaussian kernel (uncomment to use this)
    % K = @(x) exp(-norm(x)^2);
    
    % NON-BLURRING PROCESS
    if blurring ~= 1
        % Stores sample mean with kernel K for each t in T (cluster centre)
        m = zeros(size(T));
        % Count number of iterations till convergence
        num_iter = 0;
        f = figure(fig_id);
        % Convergence criteria
        while ~isequal(m, T)
            % Only assign T to m after first loop to avoid losing
            % randomly initialised cluster centres
            if num_iter >= 1
                T = m;
            end
            
            % Calculate sample mean with kernel K for each t in T
            for t = 1:length(T)
                numer = 0;
                denom = 0;
                for s = 1:length(S)
                    numer  = numer + K(S(s, :) - T(t, :))*S(s, :);
                    denom = denom + K(S(s, :) - T(t, :));
                end
                
                m(t, :) = numer/denom;
            end
            
            % Record number of loops
            num_iter = num_iter + 1
            
            % Plot cluster centres with x
            hold on
            plot(S(1:200, 1), S(1:200, 2), "+")
            plot(S(201:end, 1), S(201:end, 2), "o");
            if length(S) > 400
                plot(S(401:end, 1), S(401:end, 2), "*");
            end
            for i = 1:k
                scatter(T(i, 1), T(i, 2), 600, "k", "x")
            end
            xlabel("x")
            ylabel("y")
            title("Means Shift Clustering on 2-D Gaussian Dataset, " + ...
                "lambda = " + lambda)
            hold off
            pause(0.3)
            clf(f)
        end
        
        % Plot final cluster centres with x
        hold on
        plot(S(1:200, 1), S(1:200, 2), "+")
        plot(S(201:end, 1), S(201:end, 2), "o");
        if length(S) > 400
            plot(S(401:end, 1), S(401:end, 2), "*");
        end
        for i = 1:k
            scatter(T(i, 1), T(i, 2), 600, "k", "x")
        end
        xlabel("x")
        ylabel("y")
        title("Means Shift Clustering on 2-D Gaussian Dataset, " + ...
            "lambda = " + lambda)
        hold off
    end
end
