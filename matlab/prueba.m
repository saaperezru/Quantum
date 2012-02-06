%

folderroot='/home/tuareg/Documents/Projects/MedicalInterpretationFeatures/scripts/matlab';

addpath(fullfile(folderroot,'/NIFTI/'));

filepath='/home/tuareg/Documents/Projects/MedicalInterpretationFeatures/datasets/oasis/OAS1_0227_MR1_mpr_n4_anon_111_t88_masked_gfc.img';
D=load_nii(filepath);

%nii = make_nii(F3D);
view_nii(D);