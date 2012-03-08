dataset =  '/datasets/ORL/'

data = '../../../datasets/ORL/';
pattern = '*.pgm';

dataPath = '../../experiments/ORL/';
basisPath = '../../experiments/ORL/basis/';
LPath = '../../experiments/ORL/l/';
IPath = '../../experiments/ORL/i/';


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
