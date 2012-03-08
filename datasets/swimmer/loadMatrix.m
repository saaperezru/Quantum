function [X dim] = loadMatrix(path,pattern)

    tmp = load(path);
    Y = tmp.Y;
    num_files = size(Y,3);
    dim = [size(Y,1),size(Y,2)];
    X = zeros(dim(1)*dim(2),num_files);
    for f = 1:num_files
        img=Y(:,:,f);
        X(:,f) = img(:);
    end
end
