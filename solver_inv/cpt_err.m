function r_err_L2 = cpt_err(u_re,u_gt)
u_re = u_re(:); u_gt = u_gt(:);
k = (u_re' * u_gt) / (u_re' * u_re);
r_err_L2 = norm((k * u_re) - u_gt) / norm(u_gt);
end