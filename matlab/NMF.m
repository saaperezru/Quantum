function [basis,rep,Xrec,L,I] = NMF(X,r,path)


    WPath = fullfile(path,'W.mat');
    HPath = fullfile(path,'H.mat');
    if exist(WPath,'file') && exist(HPath,'file') 
        disp 'Loading existing matrices for NMF'
        tmp = load(WPath);
        W = tmp.W;
        tmp = load(HPath);
        H = tmp.H;
    else
   
        addpath('/home/saaperezru/QLSA/scripts/matlab/nmflib/');

        [W,H,err,v] = nmf_kl(X,r);
        save(WPath,'W');
        save(HPath,'H');
        Xrec = W*H;

        basis = W.*W;
        rep = H.*H;

        L = basis*rep;
        I = (Xrec.*Xrec)-L;

    end

end
