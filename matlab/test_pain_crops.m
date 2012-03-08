dataset =  '/datasets/pain_crops/'

data = '../../../datasets/pain_crops/';
pattern = '*.jpg';

dataPath = '../../experiments/pain_crops/';
basisPath = '../../experiments/pain_crops/basis/';
LPath = '../../experiments/pain_crops/l/';
IPath = '../../experiments/pain_crops/i/';


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
