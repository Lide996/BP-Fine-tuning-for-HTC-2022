function u = backward(A, sig, n)
u = reshape(A'*sig(:),n,n);
end