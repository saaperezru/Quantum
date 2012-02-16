%

folderroot = '/home/saaperezru/QLSA/datasets/oasis';

addpath(fullfile(folderroot,'/NIFTI/'));

filepath='../../../datasets/oasis/OAS1_0300_MR1_mpr_n4_anon_111_t88_masked_gfc.img';
D=load_nii(filepath);

%nii = make_nii(F3D);
%view_nii(D);

img = D.img;
