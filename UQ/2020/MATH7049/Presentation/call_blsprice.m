function [price] = call_blsprice(S, K, r, q, tau, sigma) 
    % Function to return Black-Scholes call option price with an underlying
    % asset that has a continuous dividend yield.
    % Inputs:
        % S       : current price of the underlying
        % K       : strike
        % r       : risk-free rate
        % tau     : time to expiration (tau > 0)
        % sigma   : volatility of the underlying i.e. forward contract

    d1 = log(S*exp((r - q)*tau)/K)/(sigma*sqrt(tau)) + sigma*sqrt(tau)/2;
    d2 = log(S*exp((r - q)*tau)/K)/(sigma*sqrt(tau)) - sigma*sqrt(tau)/2;
    % Use normcdf to compute normal cdf
    N1 = normcdf(d1); 
    N2 = normcdf(d2);
    price = S*exp(-q*tau)*N1 - K*exp(-r*tau)*N2;
end
