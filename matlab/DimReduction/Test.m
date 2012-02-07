images = '../../../datasets/oasis/';
basisPath = '../../../experimets/QLSA/basis/';
dataPath = '../../../experimets/QLSA/';
r = 5;
X = loadMatrix(images);
[B,R,Xh] = QLSA(X,r,dataPath);
clear R;
clear Xh;
basisViewer(B,basisPath,176,208,176);
