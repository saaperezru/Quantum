import numpy as np
import scipy.io as sio
from os.path import join
import os
import Control

def createBasisViewer(basisDirectory,htmlPath,Reducer,QLSA):
  try:
    os.mkdir(basisDirectory)
  except:
    print "Error Creating folder"
  return  Control.imageViewGenerator(Reducer.reduction.basisM.max(),Reducer.reduction.basisM.min(),basisDirectory,htmlPath,32,True,QLSA)
def createReconstructionViewer(imagesDirectory,htmlPath,Reducer,QLSA): 
  try:
    os.mkdir(imagesDirectory)
  except:
    print "Error Creating folder"
  return  Control.imageViewGenerator(Reducer.reduction.reconstructed.max(),Reducer.reduction.reconstructed.min(),imagesDirectory,htmlPath,32,True,QLSA)

def createRepresentationViewer(imagesDirectory,htmlPath,Reducer):
  try:
    os.mkdir(imagesDirectory)
  except:
    print "Error Creating folder"
  return  Control.imageScaledViewGenerator(Reducer.reduction.objectsM.max(),Reducer.reduction.objectsM.min(),imagesDirectory,htmlPath,1,10)

def createOriginalViewer(imagesDirectory,htmlPath,Reducer,QLSA):
  try:
    os.mkdir(imagesDirectory)
  except:
    print "Error Creating folder"
  # he maximum in the swimmer matrix is 39
  return  Control.imageOriginalViewGenerator(39.0,imagesDirectory,htmlPath,32,True,QLSA)

def buildMatrix(M,i):
    ret = np.zeros((1024,i))
    for x in range(i):
        ret[:,x]=np.asarray(M[:,:,x]).reshape(-1)
    return ret


def generateFactorization(name,method,M,p,r,QLSA=False):
  
  path = join(p,name)
  Reducer  = Control.Reducer(method,M,r,path)
  html = Control.HTMLBasisView(Reducer,path,createBasisViewer(join(path,"basisImages"),path,Reducer,QLSA))
  html.generate()
  origImgP = join(path,"originalImages")
  repImgP = join(path,"representationImages")
  recImgP = join(path,"reconstructionImages")
  htmlPath = join(path,"objects")
  html = Control.HTMLObjectsView(Reducer,path,createOriginalViewer(origImgP,htmlPath,Reducer,QLSA),createRepresentationViewer(repImgP,htmlPath,Reducer),createReconstructionViewer(recImgP,htmlPath,Reducer,QLSA))
  html.generate()
  del Reducer
  del html


I = "/home/saaperezru/datasets/swimmer/Y.mat"
tmp = sio.loadmat(I)['Y']
print "[DEBUG] Y shape: ", tmp.shape
M = buildMatrix(tmp,256)
#for i in range(M.shape[0]):
  #print M[i,0],

p = "../../experimentsPython/swimmer8"
r = 8
#generateFactorization("QLSA",Control.QLSA,M,p,r)
generateFactorization("QLSA2",Control.QLSA2,M,p,r,True)
generateFactorization("NMF",Control.NMF,M,p,r)
#generateFactorization("VQ",Control.VQ,M,p,r)
generateFactorization("PCA",Control.PCA,M,p,r)

p = "../../experimentsPython/swimmer16"
r = 16
#generateFactorization("QLSA",Control.QLSA,M,p,r)
generateFactorization("QLSA2",Control.QLSA2,M,p,r,True)
generateFactorization("NMF",Control.NMF,M,p,r)
#generateFactorization("VQ",Control.VQ,M,p,r)
generateFactorization("PCA",Control.PCA,M,p,r)
#
p = "../../experimentsPython/swimmer32"
r = 32
#generateFactorization("QLSA",Control.QLSA,M,p,r)
generateFactorization("QLSA2",Control.QLSA2,M,p,r,True)
generateFactorization("NMF",Control.NMF,M,p,r)
#generateFactorization("VQ",Control.VQ,M,p,r)
generateFactorization("PCA",Control.PCA,M,p,r)
#
p = "../../experimentsPython/swimmer100"
r = 100
generateFactorization("PCA",Control.PCA,M,p,r)
generateFactorization("QLSA2",Control.QLSA2,M,p,r)
generateFactorization("NMF",Control.NMF,M,p,r)
