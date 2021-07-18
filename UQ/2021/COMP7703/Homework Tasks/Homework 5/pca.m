function [z, W, lambdas] = pca(x)
    % Function that implements Principal Component Analysis (PCA)
    % Inputs:
        % x = dataset to perform PCA on
    % Outputs:
        % z = linear projection of x using z = W^T * (x - m)
        % W = principal components (eigenvectors of sample covariance
            % matrix) ordered according to lambdas (see below)
        % lambdas = eigenvalues of sample covariance matrix in descending
            % order
    
    % Calculating eigenvectors and eigenvalues
    % dim(W) = d x d, if choosing k columns --> d x k
    [W, lambdas] = eig(cov(x - mean(x)));
    
    % Sorting eigenvalues from largest to smallest:
    % Decomposing diagonal matrix into array of diagonals
    lambdas = diag(lambdas);
    [lambdas, idx] = sort(lambdas, 'descend');
    
    % Sorting eigenvectors (columns) according to index sequence from
    % sorting eigenvalues
    W = W(:, idx);
    
    % z = (W^T * (x - m)^T)^T = (x - m) * W modified for linear projection
    % applied to whole dataset rather than single samples.
    z = (x - mean(x))*W;
end
