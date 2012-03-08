dataset =  '/datasets/swimmer/'

data = '../../../datasets/swimmer/Y.mat';
pattern = '';

dataPath = '../../experiments/swimmer/';
basisPath = '../../experiments/swimmer/basis/';
LPath = '../../experiments/swimmer/l/';
IPath = '../../experiments/swimmer/i/';


folderroot='/home/saaperezru/QLSA/scripts';
addpath(fullfile(folderroot,dataset));

[X dim] = loadMatrix(data,pattern);
size(X)
r = 20;
[B,R,Xh,L,S] = QLSA(X,r,dataPath);
clear R;
clear Xh;
viewer(B,basisPath,'b',dim);
viewer(L,LPath,'L',dim);
viewer(S,IPath,'I',dim);
