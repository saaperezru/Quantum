dataset =  '/datasets/complex_back/'

data = '../../../datasets/complex_back/';
pattern = 'airport*.*';

dataPath = '../../experiments/complex_back_airport15/QLSA';

basisPath = '../../experiments/complex_back_airport15/QLSA/basis/';
LPath = '../../experiments/complex_back_airport15/QLSA/l/';
IPath = '../../experiments/complex_back_airport15/QLSA/i/';
recPath = '../../experiments/complex_back_airport15/QLSA/rec/';

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


dataPath = '../../experiments/complex_back_airport15/NMF';
basisPath = '../../experiments/complex_back_airport15/NMF/basis/';
LPath = '../../experiments/complex_back_airport15/NMF/l/';
IPath = '../../experiments/complex_back_airport15/NMF/i/';
recPath = '../../experiments/complex_back_airport15/NMF/rec/';

X = X+0.001;
[B,R,Xh,L,S] = NMF(X,r,dataPath);
clear R;
viewer(B,basisPath,'b',dim);
viewer(L,LPath,'L',dim);
viewer(S,IPath,'I',dim);
viewer(Xh,recPath,'img',dim);
