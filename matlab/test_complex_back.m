dataset =  '/datasets/complex_back/'

data = '../../../datasets/complex_back/';
pattern = 'airport*.*';

dataPath = '../../experiments/complex_back_airport15/';
basisPath = '../../experiments/complex_back_airport15/basis/';
LPath = '../../experiments/complex_back_airport15/l/';
IPath = '../../experiments/complex_back_airport15/i/';
recPath = '../../experiments/complex_back_airport15/rec/';

folderroot='/home/saaperezru/QLSA/scripts';
addpath(fullfile(folderroot,dataset));

[X dim] = loadMatrix(data,pattern);
r = 15;
[B,R,Xh,L,S] = QLSA(X,r,dataPath);
clear R;
viewer(B,basisPath,'b',dim);
viewer(L,LPath,'L',dim);
viewer(S,IPath,'I',dim);
viewer(Xh,recPath,'img',dim);
