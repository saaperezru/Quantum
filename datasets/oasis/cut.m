function [x,y,z] = cut(path,pattern)
    % Finds the limits for x,y and z axis that contain in the smallest uniform possible cube all the MRI images in the path found with pattern
    addpath('/home/saaperezru/QLSA/scripts/datasets/oasis/NIFTI/');

    fnames = dir(fullfile(path, pattern));
    num_files = size(fnames,1);
    filenames = cell(num_files,1);
    x = [ Inf 0 ];
    y = [ Inf 1 ];
    z = [ Inf 1 ];

    for f = 1:num_files
        disp(fnames(f).name);
        filepath=fullfile(path,fnames(f).name);
        F = load_nii(filepath);
        D = F.img;
        %Find the x axis limits
        sD = size(D,1);
        for i = 1:sD;
            if sum(sum(sum(D(i,:,:))))>0
                if x(1)>i
                    x(1) = i;
                end
                break;
            end
        end
        for i = 0:sD-1;
            if sum(sum(sum(D(sD-i,:,:))))>0
                if x(2)<i
                    x(2) = sD-i;
                end
                break;
            end
        end
        %Find the y axis limits
        sD = size(D,2);
        for i = 1:sD;
            if sum(sum(sum(D(:,i,:))))>0
                if y(1)>i
                    y(1) = i;
                end
                break;
            end
        end
        for i = 0:sD-1;
            if sum(sum(sum(D(:,sD-i,:))))>0
                if y(2)<i
                    y(2) = sD-i;
                end
                break;
            end
        end
        %Find the z axis limits
        sD = size(D,3);
        for i = 1:sD;
            if sum(sum(sum(D(:,:,i))))>0
                if z(1)>i
                    z(1) = i;
                end
                break;
            end
        end
        for i = 0:sD-1;
            if sum(sum(sum(D(:,:,sD-i))))>0
                if z(2)<i
                    z(2) = sD-i;
                end
                break;
            end
        end

    end
end
