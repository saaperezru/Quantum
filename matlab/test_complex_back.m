dataset =  '/datasets/complex_back/'

data = '../../../datasets/complex_back/';
pattern = 'airport*.*';

dataPath = '../../experiments/complex_back_airport2/';
basisPath = '../../experiments/complex_back_airport2/basis/';
LPath = '../../experiments/complex_back_airport2/l/';
IPath = '../../experiments/complex_back_airport2/i/';


folderroot='/home/saaperezru/QLSA/scripts';
addpath(fullfile(folderroot,dataset));

[X dim] = loadMatrix(data,pattern);
r = 1;
[B,R,Xh,L,S] = QLSA(X,r,dataPath);
clear R;
clear Xh;
viewer(B,basisPath,'b',dim);
viewer(L,LPath,'L',dim);
viewer(S,IPath,'I',dim);
