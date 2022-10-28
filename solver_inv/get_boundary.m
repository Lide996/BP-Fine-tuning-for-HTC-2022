function boundary = get_boundary(temp_img)
[m,n] = size(temp_img);
boundary = zeros(m,n);
for j = 2:m-1
    for k = 2:n-1
        temp_patch = temp_img(j-1:j+1,k-1:k+1);
        temp_patch = temp_patch(:);
        if max(temp_patch) ~= min(temp_patch)
            boundary(j,k) = 1;
        end
    end
end
end