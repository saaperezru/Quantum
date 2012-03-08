function [X dim] = loadMatrix(path,pattern)
    %Generates an N-by-M matrix that describes all the objects in path whose filename match the providad pattern, where N is the number of objects and M is the number of features encountered,
    % Parameters
    %   path -- Path in which the files that describe the objects can be found.
    %   pattern -- Pattern to match against filenames in the path directory
    % Returns:
    %   X -- An N-by-M matrix that describes all the objects in path whose filename match the providad pattern, where N is the number of objects and M is the number of features encountered
    %   dim -- An array with the dimensions of the original objects. This will be used by the viewer to reconstruct the objects.
    fnames = dir(fullfile(path, pattern));
    num_files = size(fnames,1);
    filenames = cell(num_files,1);

    filepath=fullfile(path,fnames(1).name);
    tmp = imread(filepath);

    dim = size(tmp);
    X =zeros(dim(1)*dim(2),num_files);

    for f = 1:num_files
        disp(fnames(f).name)
        filepath=fullfile(path,fnames(f).name);
        obj = rgb2gray(imread(filepath));
        X(:,f) = obj(:);
    end
end
