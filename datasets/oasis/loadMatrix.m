function [X dim] = loadMatrix(path,pattern)
    addpath('/home/saaperezru/QLSA/scripts/datasets/oasis/NIFTI/');

    fnames = dir(fullfile(path, pattern));
    num_files = size(fnames,1);
    

    [x y z] = cut(path,pattern);
    r = 1.0;
    % Size of the 3d image after cut and after downscaling
    xDim = ceil(((x(2)-x(1)+1)-(r/2))/r);
    yDim = ceil(((y(2)-y(1)+1)-(r/2))/r);
    zDim = ceil(((z(2)-z(1)+1)-(r/2))/r);
    dim = [xDim yDim zDim];
    
    linearDimension = dim(1)*dim(2)*dim(3);
    X = zeros(linearDimension,num_files);
    
    
    xDim = x(1):x(2);
    yDim = y(1):y(2);
    zDim = z(1):z(2);

    for f = 1:num_files
        disp(fnames(f).name)
        filepath=fullfile(path,fnames(f).name);
        file = load_nii(filepath);
        tmp=downscale(file.img(xDim,yDim,zDim),r,'nearest');
        X(:,f) = tmp(:);
    end
end
