import Control
from os.path import join

def reconstructImages(recon,path):
  imG = Control.basisToImage(recon.max(),recon.min())
  for i in range(recon.shape[1]):
    imG.toImage(recon[:,i],112,join(path,"s0"+str((i/10)+1)+"."+str((i+1)%10)))

p = "/home/tuareg/Documents/UNAL/7mo semestre/TMP/Reconstruction 1/"

Q = join(p,"QLSABasis/")
N = join(p,"NMFBasis/")
V = join(p,"VQBasis/")
PC = join(p,"PCABasis/")
QR = join(p,"QLSAReconstruct/")
NR = join(p,"NMFReconstruct/")
PR = join(p,"PCAReconstruct/")
R = "/home/tuareg/Documents/UNAL/7mo semestre/TMP/datasets/ORLFull/"

M = Control.imagesMatrix(R)
print "[DEBUG] Finish building matrix"
R,d,im = Control.ReduceQLSA(M,10,Q)
Rh = Control.quantumReconstruct(M,R.basisM)
reconstructImages(Rh,QR)
print "[DEBUG] Finish QLSA"
#Control.ReduceVQ(M,10,V)
#print "[DEBUG] Finish VQ"
R,d,im =  Control.ReduceNMF(M,10,N)
reconstructImages(R.reconstructed,NR)
print "[DEBUG] Finish NMF"
R,d,im = Control.ReducePCA(M,10,PC)
reconstructImages(R.reconstructed,PR)
print "[DEBUG] Finish PCA"

