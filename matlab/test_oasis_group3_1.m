dataset =  '/datasets/oasis/'

data = '../../../datasets/oasis_group3/1/';
pattern = '*.img';

dataPath = '../../experiments/oasis_group3/1/';
basisPath = '../../experiments/oasis_group3/1/basis/';
LPath = '../../experiments/oasis_group3/1/l/';
IPath = '../../experiments/oasis_group3/1/i/';


folderroot='/home/saaperezru/QLSA/scripts';
addpath(fullfile(folderroot,dataset));

[X dim] = loadMatrix(data,pattern);
size(X)
r = 8;
[B,R,Xh,L,S] = QLSA(X,r,dataPath);
clear R;
clear Xh;
viewer(B,basisPath,'b',dim);
viewer(L,LPath,'L',dim);
viewer(S,IPath,'I',dim);
