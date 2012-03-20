function [Pz] = marginalizeR(R):
    n = size(R,2)
    Pzd = R*((1/n)*eye(n))
    Pz = Pzd * ones(n,1)
