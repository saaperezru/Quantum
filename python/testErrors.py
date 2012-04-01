import factorization
import datasets.swimmer as sw
import numpy as np
from os import mkdir
from os.path import join

dataset = sw
X,dim = dataset.loadMatrix()
methods = {"PCA":factorization.PCA,"NMF":factorization.NMF,"QLSA":factorization.QLSA2}
errors = {"PCA":[],"NMF":[],"QLSA":[]}
dataPath = "/home/saaperezru/QLSA/experiments/swimmer"
Xn = X/np.dot(np.ones((X.shape[0],1)),np.dot(np.ones((1,X.shape[0])),X))
for m in methods.keys():
    try:
        mkdir(join(dataPath,m))
    except:
        print "[WARNING] Error while creating mehtod directory for" + m
    for i in range(1,20):
        path = join(dataPath,m,str(i))
        try:
            mkdir(path)
        except:
            print "[WARNING] Error while creating mehtod directory for" + m + " and numbre of factors " + str(i)

        B,R,Rec = methods[m](Xn,i,path)
        errors[m].append(np.linalg.norm(Xn-Rec))

for m in methods.keys():
    print "[INFO] Results for " + m
    print errors[m]
