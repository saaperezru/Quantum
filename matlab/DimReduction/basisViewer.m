function basisViewer(objects,path,x,y,z)
    % Generates MRI images in Analyze format for each column on objects, with width x, height y and depth z
    % Parameters:
    %    objects -- An feature x objects matrix
    %    path -- Directory path for storing generated images 

    folderroot='/home/tuareg/Documents/Projects/MedicalInterpretationFeatures/scripts/matlab';
    addpath(fullfile(folderroot,'/NIFTI/'));

    tmp = size(objects)
    for i=1:tmp(1)
       im = reshape(objects(:,i),x,y,z);
       save(make_nii(im),fullfile(path,'b'+str(i)));
    end
end
