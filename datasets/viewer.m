function viewer(objects,path,prefix,dim)
    % Generates a gray image for each column in the matrix of objects and stores it in path. The name of the saved file is formed by prefix + column_number
    % Parameters:
    %    objects -- An feature x objects matrix
    %    path -- Directory path for storing generated images 
    %    prefix -- A prefix for the names of the files to be generated
    %    dim -- The dimensions of the image to be generated
    tmp = size(objects);
    for i=1:tmp(2)
       %im = mat2gray(reshape(objects(:,i),dim(1),dim(2)));
       %imwrite(im,fullfile(path,strcat(prefix,num2str(i),'.bmp')));
    end
end
