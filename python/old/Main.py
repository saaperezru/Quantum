from os.path import join
import os
import Control

def createBasisViewer(basisDirectory,htmlPath,Reducer,QLSA):
  try:
    os.mkdir(basisDirectory)
  except:
    print "Error Creating folder"
  return  Control.imageViewGenerator(Reducer.reduction.basisM.max(),Reducer.reduction.basisM.min(),basisDirectory,htmlPath,112,True,QLSA)

def createReconstructionViewer(imagesDirectory,htmlPath,Reducer,QLSA):
  try:
    os.mkdir(imagesDirectory)
  except:
    print "Error Creating folder"
  return  Control.imageViewGenerator(Reducer.reduction.reconstructed.max(),Reducer.reduction.reconstructed.min(),imagesDirectory,htmlPath,112,True,QLSA)

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
  return  Control.imageOriginalViewGenerator(255,imagesDirectory,htmlPath,112,True,QLSA)

def generateFactorization(directoryName,reductionMethod,M,Docs,DocsP,p,r,QLSA=False):
 
  
  path = join(p,directoryName)
  Reducer  = Control.Reducer(reductionMethod,M,r,path,Docs,DocsP)
  html = Control.HTMLBasisView(Reducer,path,createBasisViewer(join(path,"basisImages"),path,Reducer,QLSA))
  html.generate()
  origImgP = join(path,"originalImages")
  repImgP = join(path,"representationImages")
  recImgP = join(path,"reconstructionImages")
  htmlPath = join(path,"objects")
  html = Control.HTMLObjectsView(Reducer,path,createOriginalViewer(origImgP,htmlPath,Reducer,QLSA),createRepresentationViewer(repImgP,htmlPath,Reducer),createReconstructionViewer(recImgP,htmlPath,Reducer,QLSA))
  html.generate()
  del html
  del Reducer
  

I = "/home/jecamargom/tmp/datasets/ORLFull"
M,Docs = Control.imagesMatrix(I)
DocsP = [join(I,f) for f in Docs]
print "[DEBUG] Max number in ORL faces matrix", M.max()
print "[DEBUG] Matrix Dimensions : ", M.shape

p = "/home/jecamargom/tmp/experiments/reconstruction5"
r = 5
  
#generateFactorization("QLSA",Control.QLSA,M,Docs,DocsP,p,r)
generateFactorization("QLSA2",Control.QLSA2,M,Docs,DocsP,p,r,True)
#generateFactorization("NMF",Control.NMF,M,Docs,DocsP,p,r)
#generateFactorization("VQ",Control.VQ,M,Docs,DocsP,p,r)
#generateFactorization("PCA",Control.PCA,M,Docs,DocsP,p,r)
#
p = "/home/jecamargom/tmp/experiments/reconstruction4"
r = 18
  #
#generateFactorization("QLSA",Control.QLSA,M,Docs,DocsP,p,r)
generateFactorization("QLSA2",Control.QLSA2,M,Docs,DocsP,p,r,True)
#generateFactorization("NMF",Control.NMF,M,Docs,DocsP,p,r)
#generateFactorization("VQ",Control.VQ,M,Docs,DocsP,p,r)
#generateFactorization("PCA",Control.PCA,M,Docs,DocsP,p,r)
#
p = "/home/jecamargom/tmp/experiments/reconstruction3"
r = 9
  #
#generateFactorization("QLSA",Control.QLSA,M,Docs,DocsP,p,r)
generateFactorization("QLSA2",Control.QLSA2,M,Docs,DocsP,p,r,True)
#generateFactorization("NMF",Control.NMF,M,Docs,DocsP,p,r)
#generateFactorization("VQ",Control.VQ,M,Docs,DocsP,p,r)
#generateFactorization("PCA",Control.PCA,M,Docs,DocsP,p,r)
#
