function [basis,rep,Xh] = LSA(X,r,path)
    UPath = fullfile(path,'U.mat')
    RPath = fullfile(path,'S.mat')
    VPath = fullfile(path,'V.mat')
    if exist(UPath,'file') && exist(SPath,'file') && exist(VPath,'file')
        tmp = load(UPath);
        U = tmp.U;
        tmp = load(SPath);
        S = tmp.S;
        tmp = load(VPath);
        V = tmp.V;
    else
        [U,S,V] = svd(Xn)
        save(BPath,'basis');
        save(RPath,'rep');
        save(XhPat,'Xh');
    end
    basis = U(i:,[1:r])
    rep = diag(S([0:r]))*V(:,[0:r])
    Xh = basis*rep
end
