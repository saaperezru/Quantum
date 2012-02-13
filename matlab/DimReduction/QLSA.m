function [basis,rep,Xh] = QLSA(X,r,path)
    % Reduces the space of X to r dimensions and returns the basis for this space, the representation of each point of X.
    % Parameters :
    %    X -- An feature x objects array
    %    r -- The desired dimensionality for the new space
    %    path -- A path for the algorithm to store its checkpoints
    BPath = fullfile(path,'B.mat');
    RPath = fullfile(path,'R.mat');
    XhPath = fullfile(path,'Xh.mat');
    if exist(BPath,'file') && exist(RPath,'file') && exist(XhPath,'file')
        tmp = load(BPath);
        basis = tmp.basis;
        tmp = load(RPath);
        rep = tmp.rep;
        tmp = load(XhPath);
        Xh = tmp.Xh;
    else
        % We begin by normalizing the original matrix X
       % We begin by normalizing the original matrix X
        Xsize = size(X);
        Xn = X./(ones(Xsize(1),1)*(ones(1,Xsize(1))*X));
        Xn = sqrt(Xn);
        %We need the SVD decomposition of the normalized matrix
        [B,R,Xh] = LSA(X,r,path);
        clear Xn;
        %Lets normalize the reconstruction
        %The projector over the space defined by the base B is Ps = BB^T
        Xh = normc(B*(B'*Xh));
        Xh = Xh.*Xh;
        %Lets normalize the representation
        %We want rep_{ij} to be <psi^hat_{i}|sigma_{k}>^2, which is equals to P(Z_{k}|d_{i}) : probability of latent topic Z_{k} given document i
        % And we know that <psi^hat_{i}|sigma_{k}> = <sigma_{k}|psi^hat_{i}> because both vectors have real values and so rep = (Uh.T*X)^2
        R = B'*X;
        rep = R.*R;
        clear R;
        %Lets square the basis to give it a probabilistic interpretation
        basis = B.*B;
        save(BPath,'basis');
        save(RPath,'rep');
        save(XhPath,'Xh');
    end

end
