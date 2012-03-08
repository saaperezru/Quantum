dataset =  '/datasets/cbcl/'

data = '../../../datasets/cbcl/';
pattern = '*.pgm';

dataPath = '../../experiments/cbcl/';
basisPath = '../../experiments/cbcl/basis/';
LPath = '../../experiments/cbcl/l/';
IPath = '../../experiments/cbcl/i/';


folderroot='/home/saaperezru/QLSA/scripts';
addpath(fullfile(folderroot,dataset));

[X dim] = loadMatrix(data,pattern);
size(X)
r = 40;
[B,R,Xh,L,S] = QLSA(X,r,dataPath);
clear R;
clear Xh;
viewer(B,basisPath,'b',dim);
viewer(L,LPath,'L',dim);
viewer(S,IPath,'I',dim);
