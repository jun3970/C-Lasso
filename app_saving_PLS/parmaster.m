parpool('local', 4)
% Liangjun Su, Zhentao Shi, Peter Phillips (2015)
% the master file of PLS estimation in the savings rate application

clear
global p K_max

% cvx_solver mosek

IC_needed = 1;
tol = 0.0001;
R = 80;

load('CH3-2017-09-30.mat')

X = [mkt_rf, SMB, VMG];
y = Dret_rf;

% Note: Aug 4, 2018
% A researcher asks about the time series initial value of `lagsaving` for each country.
% The initial comes from the raw data that is longer than the compiled data for the regression,
% which is not contained in `saving`

% d = size(X) returns the sizes of each dimension of array X in a vector d with ndims(X) elements. 
% [m,n] = size(X) returns the size of matrix X in separate variables m and n.
% m = size(X, dim) returns the size of the dimension of X specified by scalar dim.
p = size(X, 2);
% L = length( X )
% returns the length of the largest array dimension in X . For vectors, the length is simply the number of elements. For arrays with more dimensions, the length is max(size(X))
N = length(stkcd);
T = size(X, 1)./N;

K_max = 4;
lamb.grid = 10;
lamb.min  = 0.2;
lamb.max  = 2.0;
% the constant for lambda. very important!!
lamb_const = lamb.min * (lamb.max / lamb.min ).^( ( (1:lamb.grid) - 1) /( lamb.grid -1 ) ); 
numlam = length(lamb_const);

index = dataset( code, date, y, X );
index.Properties.VarNames = {'N'  'T'  'y'  'X'};

y_raw = y;
X_raw = X;

for i = 1:N
    % Subtract the column mean from the corresponding column elements of a matrix A. Then normalize by the standard deviation.
    % A = [1 2 10; 3 4 20; 9 6 15];
    % C = bsxfun(@minus, A, mean(A));
    % D = bsxfun(@rdivide, C, std(A))
    yi = y(index.N == i);
    %% C = bsxfun(fun,A,B) 
    % applies the element-wise binary operation specified by the function handle fun to arrays A and B.
    y(index.N == i) = bsxfun(@minus, yi, mean(yi) );  
    
    Xi = X(index.N == i, : );
    Xi = bsxfun(@minus, Xi, mean(Xi) );
    %% S = std(A)
    % If A is a vector of observations, then the standard deviation is a scalar.
    % If A is a matrix whose columns are random variables and whose rows are observations, then S is a row vector containing the standard deviations corresponding to each column.
    % If A is a multidimensional array, then std(A) operates along the first array dimension whose size does not equal 1, treating the elements as vectors. The size of this dimension becomes 1 while the sizes of all other dimensions remain the same.
    % By default, the standard deviation is normalized by N-1, where N is the number of observations.
    % S = std(A,w) specifies a weighting scheme for any of the previous syntaxes. When w = 0 (default), S is normalized by N-1. When w = 1, S is normalized by the number of observations, N. w also can be a weight vector containing nonnegative elements. In this case, the length of w must equal the length of the dimension over which std is operating.
    %% repmat(M, a, b)
    % Repeat copies of the matrix, M, into a a-by-b block arrangement
    % X(index.N == i, :) = Xi./repmat( std(Xi, 1), [T 1] ) ;
    % X_raw(index.N == i, :) = X(index.N == i, :) + repmat( mean(Xi), [T 1]);
end

ds = dataset( code, date, y, X, y_raw, X_raw );
ds.Properties.VarNames = {'N'  'T'  'y'  'X' 'y_raw' 'X_raw'};
%% initial values
beta_hat0 = zeros(N, p);
for i = 1:N
    yi = ds.y(ds.N == i );
    Xi = ds.X(ds.N == i, : );
    beta_hat0(i,:) = regress( yi , Xi );
end



%% estimation
TT = T;
IC_total = ones(K_max, numlam );

if IC_needed == 1
    for ll = 1:numlam
        disp(ll)
        
        a = ds.X \ ds.y; 
        bias = SPJ_PLS(T, ds.y_raw, ds.X_raw);
        a_corr = 2 * a - bias;
        IC_total(1, :) = mean( ( y - X*a_corr ).^2 );
        
        
        for K = 2:K_max
            Q = 999*zeros(K,1);
            
            % the parameter lambda, be related to the variance of development variable
            % and the lenght of time periods
            lam = lamb_const(ll)*var(y) * T^(-1/3); 
            [b_K, hat.a] = PLS_est(N, TT, y, X, beta_hat0, K, lam, R, tol); % estimation
            [~, H.b, ~, group] = report_b( b_K, hat.a, K );
            sum(group)            

            post_b = zeros(N, p);
            post_a = zeros(K, p);
            if K >=2
                parfor i = 1:K
                    NN = 1:N;
                    H.group = logical(group);
                    this_group = group(:,i);
                    if sum(this_group) > 0
                        g_index = NN(this_group);
                        g_data = ds( ismember(ds.N, g_index), : );

                        post = post_est_PLS_dynamic(T, g_data);
                        
                        e = g_data.y - g_data.X * post.post_a_corr ;
                        Q(i) = sum( e.^2 );
                        post_b(this_group,:) = repmat(post.post_a_corr', [sum(this_group), 1] );
                    end
                end
            end
                
            IC_total(K , ll) = sum(Q) / (N*T)
            
        end
    end
    %% calculate the IC
    pen = 2/3 * (N*T)^(-.5) * p .* repmat( (1:K_max)', [1 numlam]);
    IC_final = log(IC_total) + pen;
    disp(IC_final)
end

delete(gcp('nocreate'))

minimum = min(min(IC_final));
[K_num, lamb_index] = find(A == minimum)


%% PLS estimation
K = K_num;
lam =  lamb_const(lamb_index)*var(y) * T^(-1/3);

[b_K, a] = PLS_est(N, T, y, X, beta_hat0, K, lam, R, tol);
[~, b, ~ , group] = report_b( b_K, a, K );

%% post estimation
colnum = K*3
est_lasso = zeros(p, colnum);
est_post_lasso = zeros(p, colnum);

for i = 1:K
    NN = 1:N;
    group = logical(group);
    this_group = group(:,i);
    g_index = NN(this_group);
    g_data = ds( ismember(ds.N, g_index), : ); % group-specific data
    post = post_est_PLS_dynamic(T, g_data);
    est_post_lasso(:,(3*i-2):(3*i)) =  [post.post_a_corr, post.se, post.test_b];
end


save CH3-2017-09-30_result.mat


% %% display the estimates
% est_post_lasso = mat2dataset( est_post_lasso, 'VarNames', ...
%     {'g1_coef', 'g1_sd', 'g1_t', 'g2_coef', 'g2_sd', 'g2_t'});
% disp(est_post_lasso)
% 
% load('country56.mat')
% country(group(:,1))
% country(group(:,2))
% 
% g_PLS = zeros(56,1);
% g_PLS( group(:,1) == 1 ) = 1;
% g_PLS( group(:,2) == 1 ) = 2;
% 
% load('group_PGMM.mat')
% g_PGMM = zeros(56,1);
% g_PGMM( group_PGMM(:,2) == 1) = 2;
% g_PGMM( group_PGMM(:,1) == 1) = 1;
% 
% sum(g_PLS == g_PGMM)
% %% common FE
% 
% g_index = NN;
% first_none_zero = min( NN );
% g_data = ds( ismember(ds.N, g_index), : ); % group-specific data
% post = post_est_PLS_dynamic(T, g_data);
% 
% [post.post_a_corr, post.se, post.test_b]

