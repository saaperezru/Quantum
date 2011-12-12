import Control

Q = "/home/tuareg/Documents/UNAL/7mo semestre/TMP/datasets/QLSABasis/"
N = "/home/tuareg/Documents/UNAL/7mo semestre/TMP/datasets/NMFBasis/"
V = "/home/tuareg/Documents/UNAL/7mo semestre/TMP/datasets/VQBasis/"
P = "/home/tuareg/Documents/UNAL/7mo semestre/TMP/datasets/PCABasis/"
R = "/home/tuareg/Documents/UNAL/7mo semestre/TMP/datasets/ORLFull/"

M = Control.imagesMatrix(R)
print "[DEBUG] Finish building matrix"
#Control.ReduceQLSA(M,30,Q)
print "[DEBUG] Finish QLSA"
Control.ReduceVQ(M,30,V)
print "[DEBUG] Finish VQ"
#Control.ReduceNMF(M,30,N)
#print "[DEBUG] Finish NMF"
#Control.ReducePCA(M,30,P)
#print "[DEBUG] Finish PCA"

