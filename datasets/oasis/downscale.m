function newImage = downscale(image,percentage,method)
    if nargin < 3
        method = 'cubic';
    end
    if percentage == 1
        newImage = image;
        return
    end
    [x,y,z] = meshgrid(1/2:size(image,2),1/2:size(image,1),1/2:size(image,3));
    r=percentage;
    [xi,yi,zi] = meshgrid(r/2:r:size(image,2),r/2:r:size(image,1),r/2:r:size(image,3));
    newImage = interp3(x,y,z,image,xi,yi,zi,method);
end
