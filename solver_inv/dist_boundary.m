function Record_boundary = dist_boundary(boundary, num_layer)
[m,n] = size(boundary);
Dist_boundary = inf * ones(m,n);
Dist_boundary(boundary == 1) = 1;
Record_boundary = Dist_boundary;

diag_dist = sqrt(2);
Dist_patch = [diag_dist,1,diag_dist; 1,0,1; diag_dist,1,diag_dist];

for num_append = 1: num_layer
    for j = 2:m-1
        for k = 2:n-1
            temp_value = Dist_boundary(j,k);
            if temp_value == inf, continue, end
            Record_boundary(j-1:j+1,k-1:k+1) = min(Record_boundary(j-1:j+1,k-1:k+1), Dist_patch + temp_value);
        end
    end
    Dist_boundary = Record_boundary;
end
Record_boundary(Record_boundary == inf) = -inf;
Record_boundary(Record_boundary == -inf) = max(Record_boundary(:));
end