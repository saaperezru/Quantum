function cutImages(path,pattern,savePath)
    addpath('/home/saaperezru/QLSA/datasets/oasis/NIFTI/');

    fnames = dir(fullfile(path, pattern));
    num_files = size(fnames,1);
    

    [x y z] = cut(path,pattern);
    
    xDim = x(1):x(2);
    yDim = y(1):y(2);
    zDim = z(1):z(2);

    for f = 1:num_files
        disp(fnames(f).name);
        filepath=fullfile(path,fnames(f).name);
        saveFilepath = fullfile(savePath,fnames(f).name);
        file = load_nii(filepath);
        save_nii(make_nii(file.img(xDim,yDim,zDim)),saveFilepath);
    end
end
