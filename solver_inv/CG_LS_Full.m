function [x, r_err] = CG_LS_Full(A,weight_full,init_guess, RHS, max_iter, max_r_tol, W, sub_Sig)
n = size(init_guess,1);
% CG
init_guess = init_guess(:); RHS = RHS(:);
nb = norm(RHS); 
r_err = NaN * zeros(1,max_iter); 
Data_r_err = NaN * zeros(1,max_iter);
x = init_guess;
r = RHS - Gram_A(x);
r_err(1) = norm(r)/nb;
rr = r' * r; p = r;
if r_err(1) >= max_r_tol
    for k = 2:max_iter
        Data_r_err(k) = cpt_err(A * x,sub_Sig(:));
        Ap = Gram_A(p);
        pAp = p' * Ap;
        alpha = rr / pAp;
        x_new = x + alpha * p;
        r_new = r - alpha * Ap;
        r_err(k) =  norm(r_new)/nb;
        if r_err(k) < max_r_tol || Data_r_err(k) >= Data_r_err(k-1)
            x = x_new;
            fprintf('CG stops at %d of %d, sufficient decrease = %d, Data_r_err = %.2f%%.\n',...
                    k, max_iter,Data_r_err(k) >= Data_r_err(k-1), 100*cpt_err(A * x,sub_Sig(:)));
            break
        end
        rr_new = r_new' * r_new;
        beta = rr_new / rr;
        p = r_new + beta * p;
        x = x_new;
        r = r_new;
        rr = rr_new;
    end
end
x = reshape(x,n,n);

function y = Gram_A(x)
    y = (A' * A) * x + W .* x;
end

end