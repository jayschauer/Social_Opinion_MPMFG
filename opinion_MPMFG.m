%% Attempted implementation of the multi-population mean field game (MPMFG)

% Attempted recreation of the MPMFG algorithm from this paper:

% R. A. Banez, H. Gao, L. Li, C. Yang, Z. Han and H. V. Poor,
% "Modeling and Analysis of Opinion Dynamics in Social Networks Using
% Multiple-Population Mean Field Games," in IEEE Transactions on Signal and
% Information Processing over Networks, vol. 8, pp. 301-316, 2022,
% doi: 10.1109/TSIPN.2022.3166102.

% Please see the PDF in this repository that helps explain the below code.
% Organization:
% 1) General algorithm parameters
% 2) Social network parameters/constants
% 3) Finite difference algorithm for MPMFG
% 4) General helper functions
% 5) Social network specific helper functions

%  -------------------------------
%% 1) General algorithm parameters
%  -------------------------------
% Parameters that would be used in an arbitrary MPMFG

global max_iterations min_error w P N X_max T_max L M dx dt base_m base_v base_u;

% Stopping conditions - iterations and 'error'
max_iterations = 200; % stop after at most max_iterations
min_error = 1e-6; % or stop if error is less than min_error
% Gradient descent learning rate parameter
w = 1000; 
% There are two populations
P = 2;
N=[0.5, 0.5]; % population ratios
% [0, X_max] x [0, T_max] grid
X_max = 1;
T_max = 1;
% Grid resolution
L=50; % j = 0, ..., L => L+1 entries
M=50; % k = 0, ..., M => M+1 entries
% Space step and time_step
dx = X_max/L;
dt = T_max/M;
% Grid variables: mean field, value function, control
% Access these through the helper functions
base_m = zeros(P, L+1, M+1);
base_v = zeros(P, L+1, M+1);
base_u = zeros(P, L+1, M+1);
% Initial value of mean field is done below, before running the algorithm.
% Helper function is used to specify the initial distribution, also below.

% NOTE: I want to keep the same indices as the paper, so I have helper
% functions for accessing the matrices. These also prevent any
% out-of-bounds errors.

%  --------------------------------------
%% 2) Social network parameters/constants
%  --------------------------------------
% Parameters that are specific to the social network opinion problem

global avg_x0 lambda avg_adj adj sigma b c1 c2 c3 c4 desired_x;

% Each population's initial average opinion value
avg_x0 = [0.25, 0.75];
% Each population's stubborness
lambda = [0.9, 0.9];

% Opinion dynamics equation parameters
avg_adj = 0.001; % a_pl and a_p in the paper (with bar above a)
adj = -0.001; % a_p in the paper (without bar above a)
sigma = [0.01, 0.01];
b = [1, 1]; % Control effort constant

% Cost function parameters
c1 = 0.5;
c2 = 1;
c3 = 2;
c4 = 1;
desired_x = [0.25, 0.75]; % desired opinion

%  ----------------------------------------
%% 3) Finite difference algorithm for MPMFG
%  ----------------------------------------

% Initialize grid variables:
% Mean field gets initialized to m_0
for p = 1:1:P % inclusive range
    for j = 0:1:L
        base_m(p,(j)+1,(0)+1) = m_0(p,j); % emphasize that indices are off by 1 on base matrices
    end
    % normalize the mean field:
    initial_sum = sum(base_m(p, :, 1));
    base_m(p, :, 1) = base_m(p, :, 1)./initial_sum;
end

base_u(:) = 0.01;
% Value function stays initialized as zero

% Run the algorithm::
for it = 1:1:max_iterations
    % update mean field
    for p = 1:1:P
        for j=0:1:L
            for k=0:1:M-1
                % emphasize that indices are off by 1 on base matrices
                base_m(p, (j)+1, (k+1)+1) = update_m(p, j, k);
            end
        end
    end
    % update value function
    for p = 1:1:P
        for j=0:1:L
            for k=M:-1:2
                % emphasize that indices are off by 1 on base matrices
                base_v(p, (j)+1, (k-1)+1) = update_v(p, j, k);
            end
        end
    end
    % update control function, check error
    error_good = true; % error_good = true if all error <= min error
    for p = 1:1:P
        for j=0:1:L
            for k=0:1:M
                [new_u, gradient] = update_u(p, j, k);
                % emphasize that indices are off by 1 on base matrices
                base_u(p, (j)+1, (k)+1) = new_u;
                error = abs(gradient);
                error_good = error_good && (error <= min_error);
            end
        end
    end
    if error_good
        break
    end
    disp(it)
end
disp('done')

% Save grid variables:

%  ---------------------------
%% 4) General helper functions
%  ---------------------------
% Helper functions that would be used in an arbitrary MPMFG

% These functions take in index and return the actual value
% Lets us not worry about converting index to value elsewhere
function retval = x(j)
    global dx;
    retval = j * dx;
end

function retval = t(k)
    global dt;
    retval = k * dt;
end

% These functions are to define what happens when you try to access
% out-of-bounds index - just return zero instead of throwing error.
% Assumes p is valid.
% Need to shift index because matlab starts at 1, and I want to keep
% indices same as in the paper.
function retval = m(p, j, k)
    global L M base_m;
    if (j < 0 || j > L) || (k < 0 || k > M)
        retval = 0;
    else
        retval = base_m(p, j+1, k+1);
    end
end
function retval = v(p, j, k)
    global L M base_v;
    if (j < 0 || j > L) || (k < 0 || k > M)
        retval = 0;
    else
        retval = base_v(p, j+1, k+1);
    end
end
function retval = u(p, j, k)
    global L M base_u;
    if (j < 0 || j > L) || (k < 0 || k > M)
        retval = 0;
    else
        retval = base_u(p, j+1, k+1);
    end
end

%  -------------------------------------------
%% 5) Social network specific helper functions
%  -------------------------------------------
% Helper functions that are specific to the social network opinion problem

% Initial value of mean field.
function retval = m_0(p, j)
    % It's given in the paper that m_1_j_0 ~ N(0.25, 0.1)
    % and m_2_j_0 ~ N(0.75, 0.1)
    switch p
        case 1
            retval = normpdf(x(j), 0.25, 0.1);
        case 2
            retval = normpdf(x(j), 0.75, 0.1);
    end
end

% Average population opinion at t_k, according to mean field distribution
function retval = avg_x(p, k)
    global avg_x0 L dx base_m;
    if 0 == k
        retval = avg_x0(p);
    else
        opinions_arr = (0:1:L).*dx; % arrayfun(@(j) x(j), 0:1:L);
        mean_field_arr = reshape(base_m(p, :, (k)+1), size(opinions_arr)); % arrayfun(@(j) m(p, j, k), 0:1:L);
        retval = sum(opinions_arr.*mean_field_arr);
    end
end

% Opinion drift function
function retval = f(p, j, k)
    global P N avg_adj lambda avg_x0 adj b;
    avg_x_arr = arrayfun(@(l) avg_x(l, k), 1:1:P); % average x for each pop
    term1 = lambda(p)*sum(N.*avg_adj.*avg_x_arr);
    term2 = (1-lambda(p))*avg_adj*avg_x0(p);
    term3 = adj*x(j);
    term4 = b(p)*u(p, j, k);
    retval = term1 + term2 + term3 + term4;
end

% Running cost function
function retval = r(p, j, k)
    global c1 c2 c3 c4 P lambda desired_x avg_x0 base_m;
    term1 = c1*(u(p,j,k))^2;
    mean_field_arr = base_m(:, (j)+1, (k)+1); % arrayfun(@(l) m(l, j, k), 1:1:P);
    term2 = sum(mean_field_arr.*c2);
    term3 = c3*lambda(p)*abs(x(j) - desired_x(p));
    term4 = c4*(1-lambda(p))*abs(x(j) - avg_x0(p));
    retval = term1 + term2 + term3 + term4;
end

%% !!! Update Functions !!!
% equation 47
function new_m = update_m(p, j, k)
    global sigma dt dx;
    phi = @(p_, j_, k_) f(p_, j_, k_)*m(p_, j_, k_);
    eps = (sigma(p)^2)/2;

    new_m = 0.5*(m(p,j+1,k)+m(p,j-1,k)) + dt*(...
        -1*(phi(p, j+1, k) - phi(p, j-1, k))/(2*dx) ...
        + eps*(m(p,j+1,k)-2*m(p,j,k)+m(p,j-1,k))/dx^2 ...
        );
end

% equation 48
function new_v = update_v(p, j, k)
    global sigma P dx dt lambda N avg_adj c2 base_v base_m;

    eps = (sigma(p)^2)/2;
    
    value_arr = arrayfun(@(l) (v(l, j-1, k)-v(l, j+1, k))/(2*dx), 1:1:P);
    % causes index error: (base_v(:, (j-1)+1, (k)+1) - base_v(:, (j+1)+1, (k)+1))./(2*dx);
    mean_field_arr = reshape(base_m(:, (j)+1, (k)+1), size(value_arr));
    % arrayfun(@(l) m(l, j, k), 1:1:P);
    summation = sum(mean_field_arr.*value_arr);
    
    new_v = 0.5*(v(p, j-1, k)+v(p, j+1, k))+dt*(...
        r(p,j,k) + m(p,j,k)*c2 - f(p,j,k)*(v(p, j-1, k)-v(p, j+1, k))/(2*dx) ...
        - lambda(p)*N(p)*avg_adj*x(j)*summation...
        - eps*(v(p,j-1,k)-2*v(p,j,k)+v(p,j+1,k))/dx^2 ...
        );
end

% equation 49
function [new_u, gradient] = update_u(p, j, k)
    global c1 w;
    gradient = 2*c1*u(p,j,k);
    new_u = (w/(1+w))*u(p,j,k) - (1/(1+w))*gradient;
end