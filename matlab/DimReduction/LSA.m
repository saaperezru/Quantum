function [basis,rep,Xh] = LSA(X,r,path)
    UPath = fullfile(path,'U.mat');
    SPath = fullfile(path,'S.mat');
    VPath = fullfile(path,'V.mat');
    if exist(UPath,'file') && exist(SPath,'file') && exist(VPath,'file')
        tmp = load(UPath);
        U = tmp.U;
        tmp = load(SPath);
        S = tmp.S;
        tmp = load(VPath);
        V = tmp.V;
    else
        [U,S,V] = svd(X);
        save(UPath,'U');
        save(SPath,'S');
        save(VPath,'V');
    end
    basis = U(:,[1:r]);
    rep = S([1:r],[1:r])*(V(:,[1:r])');
    Xh = basis*rep;
end
