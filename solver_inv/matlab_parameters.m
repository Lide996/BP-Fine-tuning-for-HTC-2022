%% Initializing b
s_b_imp = 0.05;
%% Initializing Sig_S
lambda_sb = 0.25;
sigma_d = 30 / 255;
pxd = 4; pyd = 4; sxd = 1; syd = 1;
%% Initializing d
s_d_imp = 0.025; lambda_d = 0.5; lambda_bd = 1; 
%% Initializing PsiQ
lambda_sd = 2; lambda_fd_imp = 40 / 255;
pxq = 8; pyq = 8; sxq = 1; syq = 1;
DDTF_iter = 3;
%% Iteration
num_main_loop = 3;
lambda_u_imp = 3;
lambda_d_imp = 0;
lambda_pd = 0.5;
lambda_b = 1; lambda_pb = 4;
s_u_imp = 10; mu_imp = 0.5;
s_b = lambda_b * s_b_imp^2; Main_Bregman_iter = 20; R_uv_main_tol = 5E-3;
%%
num_layer = 10;