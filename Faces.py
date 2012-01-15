from os.path import join
import os
import Control
import numpy as np

def createBasisViewer(basisDirectory,htmlPath,Reducer,QLSA):
  try:
    os.mkdir(basisDirectory)
  except:
    print "Error Creating folder"
  return  Control.imageViewGenerator(Reducer.reduction.basisM.max(),Reducer.reduction.basisM.min(),basisDirectory,htmlPath,19,True,QLSA)

def createReconstructionViewer(imagesDirectory,htmlPath,Reducer,QLSA):
  try:
    os.mkdir(imagesDirectory)
  except:
    print "Error Creating folder"
  return  Control.imageViewGenerator(Reducer.reduction.reconstructed.max(),Reducer.reduction.reconstructed.min(),imagesDirectory,htmlPath,19,True,QLSA)

def createRepresentationViewer(imagesDirectory,htmlPath,Reducer,h):
  try:
    os.mkdir(imagesDirectory)
  except:
    print "Error Creating folder"
  return  Control.imageScaledViewGenerator(Reducer.reduction.objectsM.max(),Reducer.reduction.objectsM.min(),imagesDirectory,htmlPath,h,10)

def createOriginalViewer(imagesDirectory,htmlPath,Reducer,QLSA):
  try:
    os.mkdir(imagesDirectory)
  except:
    print "Error Creating folder"
  return  Control.imageOriginalViewGenerator(255,imagesDirectory,htmlPath,19,True,QLSA)

def generateFactorization(directoryName,reductionMethod,M,Docs,DocsP,p,r,QLSA=False):
 
  
  path = join(p,directoryName)
  Reducer  = Control.Reducer(reductionMethod,M,r,path,Docs,DocsP)
  html = Control.HTMLBasisView(Reducer,path,createBasisViewer(join(path,"basisImages"),path,Reducer,QLSA))
  html.generate()
  origImgP = join(path,"originalImages")
  repImgP = join(path,"representationImages")
  recImgP = join(path,"reconstructionImages")
  htmlPath = join(path,"objects")
  html = Control.HTMLObjectsView(Reducer,path,createOriginalViewer(origImgP,htmlPath,Reducer,QLSA),createRepresentationViewer(repImgP,htmlPath,Reducer,np.ceil(np.sqrt(r))),createReconstructionViewer(recImgP,htmlPath,Reducer,QLSA))
  html.generate()
  del html
  del Reducer
  

I = "/home/jecamargom/tmp/datasets/faces"
M,Docs = Control.imagesMatrix(I,361)
DocsP = [join(I,f) for f in Docs]
print "[DEBUG] Max number in ORL faces matrix", M.max()
print "[DEBUG] Matrix Dimensions : ", M.shape
p = "/home/jecamargom/tmp/experiments/faces1"
r = 50
  
#generateFactorization("QLSA",Control.QLSA,M,Docs,DocsP,p,r)
generateFactorization("QLSA2",Control.QLSA2,M,Docs,DocsP,p,r,True)
#generateFactorization("NMF",Control.NMF,M,Docs,DocsP,p,r)
#generateFactorization("VQ",Control.VQ,M,Docs,DocsP,p,r)
#generateFactorization("PCA",Control.PCA,M,Docs,DocsP,p,r)

p = "/home/jecamargom/tmp/experiments/faces2"
r = 25
  
#generateFactorization("QLSA",Control.QLSA,M,Docs,DocsP,p,r)
generateFactorization("QLSA2",Control.QLSA2,M,Docs,DocsP,p,r,True)
#generateFactorization("NMF",Control.NMF,M,Docs,DocsP,p,r)
#generateFactorization("VQ",Control.VQ,M,Docs,DocsP,p,r)
#generateFactorization("PCA",Control.PCA,M,Docs,DocsP,p,r)

p = "/home/jecamargom/tmp/experiments/faces3"
r = 100
  
#generateFactorization("QLSA",Control.QLSA,M,Docs,DocsP,p,r)
generateFactorization("QLSA2",Control.QLSA2,M,Docs,DocsP,p,r,True)
#generateFactorization("NMF",Control.NMF,M,Docs,DocsP,p,r)
#generateFactorization("VQ",Control.VQ,M,Docs,DocsP,p,r)
#generateFactorization("PCA",Control.PCA,M,Docs,DocsP,p,r)
