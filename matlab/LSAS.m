function [basis,rep,Xh] = LSA(X,r,path)
    UPath = fullfile(path,'U.mat');
    SPath = fullfile(path,'S.mat');
    VPath = fullfile(path,'V.mat');
    if exist(UPath,'file') && exist(SPath,'file') && exist(VPath,'file')
        disp 'Loading existing matrices for SVD in LSA'
        tmp = load(UPath);
        U = tmp.U;
        tmp = load(SPath);
        S = tmp.S;
        tmp = load(VPath);
        V = tmp.V;
    else
        [U,S,V] = svds(sparse(X));
        save(UPath,'U','-v7.3');
        save(SPath,'S','-v7.3');
        save(VPath,'V','-v7.3');
    end
    basis = U(:,[1:r]);
    rep = S([1:r],[1:r])*(V(:,[1:r])');
    Xh = basis*rep;
end
