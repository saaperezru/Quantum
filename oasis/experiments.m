
    [x y z] = cut(path,pattern);
    r = 1.0;
    % Size of the 3d image after cut and after downscaling
    xDim = ceil(((x(2)-x(1)+1)-(r/2))/r);
    yDim = ceil(((y(2)-y(1)+1)-(r/2))/r);
    zDim = ceil(((z(2)-z(1)+1)-(r/2))/r);
    dim = [xDim yDim zDim];

