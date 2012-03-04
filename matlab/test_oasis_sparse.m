dataset =  '/datasets/oasis/'

data = '../../../datasets/oasis/oasis-group1';
pattern = '*.img';

dataPath = '../../experiments/oasis3/';
basisPath = '../../experiments/oasis3/basis/';
LPath = '../../experiments/oasis3/l/';
IPath = '../../experiments/oasis3/i/';


folderroot='/home/saaperezru/QLSA/scripts';
addpath(fullfile(folderroot,dataset));

[X dim] = loadMatrix(data,pattern);
size(X)
r = 5;
[B,R,Xh,L,S] = QLSA(X,r,dataPath);
clear R;
clear Xh;
viewer(B,basisPath,'b',dim);
viewer(L,LPath,'L',dim);
viewer(S,IPath,'I',dim);
