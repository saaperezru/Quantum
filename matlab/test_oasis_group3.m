dataset =  '/datasets/oasis/'

data = '../../../datasets/oasis_group3/full/';
pattern = '*.img';



folderroot='/home/saaperezru/QLSA/scripts';
addpath(fullfile(folderroot,dataset));

path = '../../experiments/oasis_group3/'

[X dim] = loadMatrix(data,pattern);
size(X)
dataPathInit = fullfile(path,strcat('full',num2str(80)));
%for i = [80]
%    dataPath = fullfile(path,strcat('full',num2str(i)));
%    basisPath = fullfile(path,strcat('full',num2str(i)),'basis');
%    LPath = fullfile(path,strcat('full',num2str(i)),'L');
%    IPath = fullfile(path,strcat('full',num2str(i)),'I');
%    mkdir(dataPath);
%    mkdir(basisPath);
%    mkdir(LPath);
%    mkdir(IPath);
%    r = i
%    [B,R,Xh,L,S] = QLSA(X,r,dataPath);
%    clear R;
%    clear Xh;
%    viewer(B,basisPath,'b',dim);
%    viewer(L,LPath,'L',dim);
%    viewer(S,IPath,'I',dim);
%end
for i = [ 20, 40]
    dataPath = fullfile(path,strcat('full',num2str(i)));
    basisPath = fullfile(path,strcat('full',num2str(i)),'basis');
    LPath = fullfile(path,strcat('full',num2str(i)),'L');
    IPath = fullfile(path,strcat('full',num2str(i)),'I');
    mkdir(dataPath);
    mkdir(basisPath);
    mkdir(LPath);
    mkdir(IPath);
    copyfile(fullfile(dataPathInit,'U.mat'),fullfile(dataPath,'U.mat'))
    copyfile(fullfile(dataPathInit,'V.mat'),fullfile(dataPath,'V.mat'))
    copyfile(fullfile(dataPathInit,'S.mat'),fullfile(dataPath,'S.mat'))
    r = i
    [B,R,Xh,L,S] = QLSA(X,r,dataPath);
    clear R;
    clear Xh;
    viewer(B,basisPath,'b',dim);
    viewer(L,LPath,'L',dim);
    viewer(S,IPath,'I',dim);
end
