function viewer(objects,path,prefix,dim)
    % Generates MRI images in Analyze format for each column on objects, with width x, height y and depth z
    % Parameters:
    %    objects -- An feature x objects matrix
    %    path -- Directory path for storing generated images 
    addpath('/home/saaperezru/QLSA/datasets/oasis/NIFTI/');

    tmp = size(objects);
    for i=1:tmp(1)
       im = reshape(objects(:,i),dim(1),dim(2),dim(3));
       save(make_nii(im),fullfile(path,strcat(prefix,num2str(i))));
    end
end
