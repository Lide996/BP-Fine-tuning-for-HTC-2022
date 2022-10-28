%% loading data
addpath('./solver_inv')
addpath(genpath('./toolbox/'))
load('./temp/tmp_result.mat')
u_hat = get_bin(double(u_hat),0.5);
matlab_parameters
n = 512;
angles = CtDataLimited.parameters.angles; 
num_meas = length(angles);
num_bin = size(CtDataLimited.sinogram,2);
%% Construct full anglular signal
A = create_ct_operator_2d_fan_astra_cuda(CtDataLimited, n, n);
%% Initialization stage
sub_Sig = CtDataLimited.sinogram; sub_Sig = sub_Sig / max(sub_Sig(:));
%% 1.1 Initializing b
b = sub_Sig; b(abs(b) < s_b_imp) = 0;
%% 1.2 Initializing u
% % % load('result.mat'); u_hat = double(albedo); clear albedo
Sig_hat = forward(A, u_hat, num_meas, num_bin);
amp_const = (Sig_hat(:)' * sub_Sig(:)) / (Sig_hat(:)' * Sig_hat(:));
u_hat = amp_const * u_hat; Sig_hat = amp_const * Sig_hat;
u = u_hat;
b_u = forward(A, u, num_meas, num_bin);
%% Iteration stage
err_b = sum((b_u(:) - b(:)).^2);
lambda_u_imp = 3; lambda_u = lambda_u_imp * err_b / sum(u_hat(:).^2);
WW = inf_nor(dist_boundary(get_boundary(u_hat), num_layer));
%% Reinitializing u
[u, ~] = CG_LS_Full(A, lambda_d + lambda_d * lambda_pd * lambda_sd, zeros(n,n),...
                backward(A, b, n) +...
                lambda_u * WW .* u_hat,...
             50, 0, lambda_u * WW(:),sub_Sig);
u = get_bin(u,0.5);
cd('./temp')
save final_result u
cd ..