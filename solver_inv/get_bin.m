function u = get_bin(u,thresh_value)
u(u < 0) = 0;
if max(u(:)) > 0, u = inf_nor(u); end
u = double(u > thresh_value);
end