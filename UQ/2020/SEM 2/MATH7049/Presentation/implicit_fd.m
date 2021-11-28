% Solves the untransformed B-S PDE for the European call option price of a
% given underlying that has a continuous dividend yield.
clear all;
format long
S0 = 100;       % Initial asset price
K = 100;        % Strike price
r = 0.02;       % Interest rate
q = 0.01;       % Dividend yield, can vary this to show stability holds!
sigma = 0.2;    % Option volatility
T = 2;          % Time to expiry
theta = 1;      % 0 for explicit, 1 for implicit, and 0.5 for CN

% Grid domain
Smin = 0;
Smax = 500; 
N_list = [5 .*2.^[0:5]];    % Time
M_list = [100 .*2.^[0:5]];  % Space 

hold on
tic
for nn = 1:length(N_list)
    N = N_list(nn);     % Number of time intervals,
    MM = M_list(nn);    % Number of space intervals

    dt = T/N;               % Time stepsize
    dS = (Smax-Smin)/MM;    % Space stepsize             
    S = Smin:dS:Smax;       % Space partition 
    
    % Compute the discretisation matrix, MM = 2*J
    m = MM - 1 ; % Number of interior points
    
    % Initial condition - call payoff only for interior points
    u = zeros(m, 1);
    for s = 2:length(S) - 1
        u(s - 1) = max(S(s) - K, 0);
    end
        
    % Use sparse function for faster runtime
    I = sparse(eye(m)); % Identify matrix of size m
    e = ones(m, 1);     % Vector of ones
        
    % spdiags(B, d, m, n) creates an m-by-n sparse matrix by taking the
    % columns of B and placing them along the diagonals specified by d, see
    % Practical 7.
        % [-e 0*e e] forms [-1 0 1] pattern.
        % [-e -2*e e] forms the [1 -2 1] pattern.
        % d = -1:1 -> tridigonal using the digonal indexing system.
    M1 = spdiags([-e 0*e e], -1:1, m, m);
    M2 = spdiags([e -2*e e], -1:1, m, m);

    % Compute the coefficient vectors, see L4.17
        % Dividing f by (dS)^2 and g by (2*dS) so don't have to worry
        % for F, G and H later.
    f = -0.5 .* sigma .^2 .* S.^2 /dS/dS;
    g = -(r - q) .* S/2/dS;
    h = r .* ones(size(S));
    
    % Form M from 3 split parts
    M = sparse(diag(f(2:end - 1))) * M2 + ...
        sparse(diag(g(2: end - 1))) * M1 + ...
        sparse(diag(h(2: end - 1))) * I;

    % Form left hand side and right hand side matrices - using second form
    % in practical 7 (/delta_tau).
    lhs_mat = I/dt + theta * M;
    rhs_mat = I/dt - (1 - theta) * M;

    umin_pre = 0;
    umax_pre = Smax;
    % Loop over timestep
    for n = 1:N
        rhs_vec = rhs_mat * u;
        
        % Use linearity boundary conditions:
        % Since we cannot have boundary conditions from linear system, need
        % to impose boundary conditions on not just current time step, but
        % also on previous time step.
        
        % Lower boundary value for call
        umin_now = 2*u(1) - u(2);           % At current time step    
        % Upper boundary values for call   
        umax_now = 2*u(end) - u(end - 1);   % At current time step
           
        % Adjust the boundary conditions = -theta*p_(n+1) - (1 - theta)*p_n
            % See L4.11
        rhs_vec(1) = rhs_vec(1) - theta*(f(1) + g(1)) * umin_now - ...
            (1 - theta) * (f(1) + g(1)) * umin_pre;
        
        rhs_vec(end) = rhs_vec(end) - theta*(f(end) - g(end)) * umax_now - ...
            (1 - theta) * (f(end) - g(end)) * umax_pre; 
            
        % lhs_mat * U = rhs_vec -> solve for U
        u = lhs_mat\rhs_vec;
        
        % Update umin_pre and umax_pre
        umin_pre = umin_now;    % At previous time step
        umax_pre = umax_now;    % At previous time step
    end
    
    % Plot results and find correct time-0 price
    plot(S(2:end - 1), u);
    price(1, nn) = u(find(abs(S(2:end - 1) -S0)<=1e-12), 1);
end
toc

plot(linspace(K, K, length(u)), u, "r:");
legend("N=5, M=100", "N=10, M=200", "N=20, M=400", "N=40, M=800", ...
    "N=80, M=1600", "N=160,M=3200");
xlabel("S_j");
ylabel("C^{Imp}_{N,M}(S_j, t_0)");
hold off

% Exact call option price
exact_call = call_blsprice(S0, K, r, q, T, sigma);
disp(sprintf("Exact call price: %.9g \n", exact_call));
error_vec = exact_call - price;

% Results table
fprintf("N \t  M \t  Price \t Error \t\t\t Ratio \n");
for nn = 1:length(N_list)
    if N_list(nn) < 100
        fprintf("%1d \t %1d \t %3.6f", N_list(nn), M_list(nn), price(nn));
    else
        fprintf("%1d  %1d \t %3.6f", N_list(nn), M_list(nn), price(nn));
    end
    
    if (nn > 0)
        fprintf("\t %1.3e  ", error_vec(nn));
    end
    
    if (nn > 1)
        ratio(nn-1) = error_vec(nn-1)/error_vec(nn);
        fprintf("\t %3.3f", ratio(nn-1));
    end
    
    fprintf("\n");
end
