function boundary = ext_boundary(temp_boundary)
[m,n] = size(temp_boundary);
boundary = zeros(m,n);
for j = 2:m-1
    for k = 2:n-1
        temp_patch = temp_boundary(j-1:j+1,k-1:k+1);
        temp_patch = temp_patch(:);
        if max(temp_patch) == 1
            boundary(j,k) = 1;
        end
    end
end
end