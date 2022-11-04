%% loading data
addpath('./solver_inv')
addpath(genpath('./toolbox/'))
load('./temp/tmp_result.mat')
u_hat = get_bin(double(u_hat),0.5);
matlab_parameters
n = 512;
num_meas = length(CtDataLimited.parameters.angles);
num_bin = size(CtDataLimited.sinogram,2);
A = create_ct_operator_2d_fan_astra_cuda(CtDataLimited, n, n);
sub_Sig = CtDataLimited.sinogram; sub_Sig = sub_Sig / max(sub_Sig(:));
%% Initializing u
Sig_hat = forward(A, u_hat, num_meas, num_bin);
amp_const = (Sig_hat(:)' * sub_Sig(:)) / (Sig_hat(:)' * Sig_hat(:));
u_hat = amp_const * u_hat; Sig_hat = amp_const * Sig_hat;
max_u_hat = max(u_hat(:));
u = u_hat; b_u = forward(A, u, num_meas, num_bin);
Upper_bound = max(u(:));
%% 1.3 Iteration
err_data = sum((b_u(:) - sub_Sig(:)).^2);
err_u = err_u_imp * sum(u(:).^2);
if err_u == 0
    fprintf('Error: u = 0!')
end
lambda_u = err_data / err_u; mu = 0.5 * lambda_u;
q = zeros(n,n);
Err_uv = NaN * zeros(1,Num_iter);
Err_data = Err_uv;
Err_hat = Err_uv;
for num_iter = 1: Num_iter
    v = u - q; v(v < 0) = 0; v(v > Upper_bound) = Upper_bound;
    u = CG_LS_Full(A, zeros(n,n), backward(A, sub_Sig, n) + lambda_u * u_hat + mu * (v + q),...
                   20, 5e-3, lambda_u + mu ,sub_Sig);
    q = q + v - u;
    Err_uv(num_iter) = norm(u(:)-v(:))/norm(u(:));
    if Err_uv(num_iter) < 5E-3
        fprintf('Err_uv < 5E-3 at iter %d\n',num_iter)
        break
    end
    Err_data(num_iter) = sum(sum((forward(A,u,num_meas,num_bin)-sub_Sig).^2));
    Err_hat(num_iter) = sum(sum((u - u_hat).^2));
end
%% Boundary
boundary = get_boundary(u_hat);
change_rate = (v - u_hat) / max(u_hat(:));
u = u_hat;
u(boundary == 1 & change_rate >= 0.1) = max_u_hat;
u(boundary == 1 & change_rate <= -0.1) = 0;
boundary = ext_boundary(boundary);
u(boundary == 1 & change_rate >= 0.3) = max_u_hat;
u(boundary == 1 & change_rate <= -0.3) = 0;
boundary = ext_boundary(boundary);
u(boundary == 1 & change_rate >= 0.5) = max_u_hat;
u(boundary == 1 & change_rate <= -0.5) = 0;
%%
u = get_bin(u,0.5);
cd('./temp')
save final_result u
cd ..