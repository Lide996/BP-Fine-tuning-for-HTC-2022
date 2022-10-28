function sig = forward(A, u, num_meas, num_bin)
sig = reshape(A * u(:), num_meas, num_bin);
end