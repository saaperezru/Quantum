function [basis,rep,Xh] = VQ(X,r,path)
    BPath = fullfile(path,'B.mat')
    RPath = fullfile(path,'R.mat')
    XhPath = fullfile(path,'Xh.mat')
    if exist(BPath,'file') && exist(RPath,'file') && exist(XhPath,'file')
        tmp = load(BPath);
        basis = tmp.basis;
        tmp = load(RPath);
        rep = tmp.rep;
        tmp = load(XhPath);
        Xh = tmp.Xh;
    else
        %kmeans receives an n-by-p matrix, n samples or objects, p variables or features
        [IDX,C] = kmeans(X',r)
        basis = C'
        tmp = size(X)
        rep = zeros(r,tmp(1))
        for i = 1:tmp(1)
            rep(IDX(i),i)=1
        Xh = basis*rep
end
