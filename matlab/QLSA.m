function [basis,rep,Xrec,L,I] = QLSA(X,r,path)
    % Reduces the space of X to r dimensions and returns the basis for this space, the representation of each point of X.
    % Parameters :
    %    X -- An feature x objects array
    %    r -- The desired dimensionality for the new space
    %    path -- A path for the algorithm to store its checkpoints
    BPath = fullfile(path,'B.mat');
    RPath = fullfile(path,'R.mat');
    XhPath = fullfile(path,'Xh.mat');
    LPath = fullfile(path,'L.mat');
    IPath = fullfile(path,'I.mat');
    if exist(BPath,'file') && exist(RPath,'file') && exist(XhPath,'file')
        tmp = load(BPath);
        basis = tmp.basis;
        tmp = load(RPath);
        rep = tmp.rep;
        tmp = load(XhPath);
        Xrec = tmp.Xrec;
        tmp = load(LPath);
        L = tmp.L;
        tmp = load(IPath);
        I = tmp.I;
    else
        % We begin by normalizing the original matrix X
        % We begin by normalizing the original matrix X
        Xsize = size(X);
        Xquan = X./(ones(Xsize(1),1)*(ones(1,Xsize(1))*X));
        Xquan = sqrt(Xquan);
        %We need the SVD decomposition of the normalized matrix
        [B,Ro,Xquanr] = LSA(Xquan,r,path);
        clear Ro;
        clear Xquan;
        %Lets normalize the reconstruction
        %The projector over the space defined by the base B is Ps = BB^T
        % Xh before normalization is psi bar, after normalization is psi hat (i.e. psi bar normalized and squared)
        % We should then project X in the obtained basis like Xh = (B*(B'*X)); but it is almost the same as doing B*R (what Xho is) because B'*X is almost the same as R (as shown below)
        %sum(Xh-Xho)
        Xquanr = normc(Xquanr);
        Xrec = Xquanr.*Xquanr;
        %Lets normalize the representation
        %We want rep_{ij} to be <psi^hat_{i}|sigma_{k}>^2, which is equals to P(Z_{k}|d_{i}) : probability of latent topic Z_{k} given document i
        % And we know that <psi^hat_{i}|sigma_{k}> = <sigma_{k}|psi^hat_{i}> because both vectors have real values and so rep = (Uh.T*X)^2
        % We could do R = B'*X; but this would be the same as taking Ro
        %sum(R-Ro)
        rep = B'*Xquanr;
        rep = rep.*rep;
        %Lets square the basis to give it a probabilistic interpretation
        basis = B.*B;
        % as PTZ = basis and PTD = Xh, then I = Xh-basis*PZD
        L = basis * rep;
        I = Xrec - L;
        save(BPath,'basis');
        save(RPath,'rep');
        save(XhPath,'Xrec');
        save(LPath,'L');
        save(IPath,'I');
    end
end
