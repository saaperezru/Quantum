dataset =  '/datasets/oasis/'

data = '../../../datasets/oasis/';
pattern = '*.img';

dataPath = '../../experiments/oasis/';
basisPath = '../../experiments/oasis/basis/';
LPath = '../../experiments/oasis/l/';
IPath = '../../experiments/oasis/i/';


folderroot='/home/saaperezru/QLSA/';
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
