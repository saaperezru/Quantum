function [X] = loadMatrix(path)
    folderroot='/home/tuareg/Documents/Projects/MedicalInterpretationFeatures/scripts/matlab';
    addpath(fullfile(folderroot,'/NIFTI/'));

    fnames = dir(fullfile(path, '*.img'));
    num_files = size(fnames,1);
    filenames = cell(num_files,1);

    filepath=fullfile(path,fnames(1).name);
    D=load_nii(filepath);
    tmp = size(D.img);
    X =zeros(tmp(1)*tmp(2)*tmp(3),num_files);

    for f = 1:num_files
        disp(fnames(f).name)
        filepath=fullfile(path,fnames(f).name);
        D=load_nii(filepath);
        X(:,f) = D.img(:);
    end
end
