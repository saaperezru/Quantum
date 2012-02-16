function [X dim] = loadMatrix(path,pattern)
    fnames = dir(fullfile(path, pattern));
    num_files = size(fnames,1);
    filenames = cell(num_files,1);

    filepath=fullfile(path,fnames(1).name);
    tmp = size(imread(filepath));
    X =zeros(tmp(1)*tmp(2),num_files);
    dim = [tmp(1) tmp(2)];

    for f = 1:num_files
        disp(fnames(f).name)
        filepath=fullfile(path,fnames(f).name);
        img=rgb2gray(imread(filepath));
        X(:,f) = img(:);
    end
end
