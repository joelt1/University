function [call, put] = my_bls_price(S, K, tau, R_grow, R_disc, sigma) 
    % Function to return generalised Black-Scholes call and put option
    % prices.
    % Inputs:
        % S       : current price of the underlying.
        % K       : strike.
        % r       : risk-free rate.
        % tau     : time to expiration (tau > 0).
        % sigma   : volatility of the underlying i.e. forward contract.

    d1 = (log(S/K) + (R_grow + (sigma^2)/2)*tau)/(sigma*sqrt(tau));
    d2 = d1 - sigma*sqrt(tau);
    % Use normcdf to compute normal cdf
    call = exp(-R_disc*tau)*(S*exp(R_grow*tau)*normcdf(d1) - ...
        K*normcdf(d2));
    put = exp(-R_disc*tau)*(K*normcdf(-d2) - ...
        S*exp(R_grow*tau)*normcdf(-d1));
end
