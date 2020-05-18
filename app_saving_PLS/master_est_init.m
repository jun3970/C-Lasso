clear
global p

load('CH3-2017-09-30.mat')
X = [mkt_rf, SMB, VMG];
y = Dret_rf;

p = size(X, 2);
N = length(stkcd);
T = size(X, 1)./N;

% parameter for convergence
tol = 0.0001; % convergence tolerance level
R = 80; %  maximum number of iterations

%% demean and de-variance (not useful for simulation but useful for empirical application)
index = dataset( code, date, y, X );
index.Properties.VarNames = {'N'  'T'  'y'  'X'};

y_raw = y;
X_raw = X;

for i = 1:N
    yi = y(index.N == i);
    y(index.N == i) = bsxfun(@minus, yi, mean(yi) );  
    
    Xi = X(index.N == i, : );
    Xi = bsxfun(@minus, Xi, mean(Xi) );
end

% prepare the dataset. Useful for the functions.
ds = dataset( code, date, y, X, y_raw, X_raw );
ds.Properties.VarNames = {'N'  'T'  'y'  'X' 'y_raw' 'X_raw'};
%% initial values
beta_hat0 = zeros(N, p);
for i = 1:N
    yi = ds.y(ds.N == i );
    Xi = ds.X(ds.N == i, : );
    beta_hat0(i,:) = regress( yi , Xi ); 
end

%%
TT = T;

% PLS estimation
K = 3;
lam = 0.4309 * var(y) * T^(-1/3);

[b_K, a] = PLS_est(N, T, y, X, beta_hat0, K, lam, R, tol);
[~, b, ~ , group] = report_b( b_K, a, K);

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


if K == 2
    est_post_lasso = mat2table(est_post_lasso, 'VariableNames', ...
        ['g1_coef', 'g1_sd', 'g1_t', 'g2_coef', 'g2_sd', 'g2_t'])
elseif K == 3        
    est_post_lasso = mat2table(est_post_lasso, 'VariableNames', ...
        ['g1_coef', 'g1_sd', 'g1_t', 'g2_coef', 'g2_sd', 'g2_t', 'g3_coef', 'g3_sd', 'g3_t'])
else
    disp('Attention! The group number have not set in this script.')
end


g_PLS = zeros(N, 1);
if K == 2
    g_PLS( group(:,1) == 1 ) = 1;
    g_PLS( group(:,2) == 1 ) = 2;
elseif K == 3
    g_PLS( group(:,1) == 1 ) = 1;
    g_PLS( group(:,2) == 1 ) = 2;
    g_PLS( group(:,3) == 1 ) = 3;
end
% Convert cell to a table and use first row as variable names
g_PLS = array2table([stkcd g_PLS], 'VariableNames', ["Stkcd", "g_PLS"]);

%% common FE
g_index = NN;
first_none_zero = min( NN );
g_data = ds( ismember(ds.N, g_index), : ); % group-specific data
post = post_est_PLS_dynamic(T, g_data);
% display the estimates
[post.post_a_corr, post.se, post.test_b]

save RESULT_2017-09-30_6_21_CH3.mat
writetable('PLS_2017-09-30_6_21_CH3.csv', est_post_lasso)
writetable('group_2017-09-30_6_21_CH3.csv', g_PLS)
