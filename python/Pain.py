import numpy as np
import scipy.io as sio
from os.path import join
import os
import Control

def createBasisViewer(basisDirectory,htmlPath,Reducer):
  try:
    os.mkdir(basisDirectory)
  except:
    print "Error Creating folder"
  return  Control.imageViewGenerator(Reducer.reduction.basisM.max(),Reducer.reduction.basisM.min(),basisDirectory,htmlPath,241)

def createReconstructionViewer(imagesDirectory,htmlPath,Reducer):
  try:
    os.mkdir(imagesDirectory)
  except:
    print "Error Creating folder"
  return  Control.imageViewGenerator(Reducer.reduction.reconstructed.max(),Reducer.reduction.reconstructed.min(),imagesDirectory,htmlPath,241)

def createRepresentationViewer(imagesDirectory,htmlPath,Reducer):
  try:
    os.mkdir(imagesDirectory)
  except:
    print "Error Creating folder"
  return  Control.imageScaledViewGenerator(Reducer.reduction.objectsM.max(),Reducer.reduction.objectsM.min(),imagesDirectory,htmlPath,4,10)

def createOriginalViewer(imagesDirectory,htmlPath,Reducer):
  try:
    os.mkdir(imagesDirectory)
  except:
    print "Error Creating folder"
  # he maximum in the swimmer matrix is 39
  return  Control.imageOriginalViewGenerator(Reducer.reduction.data.max(),imagesDirectory,htmlPath,241)



def generateFactorization(name,method,M,p,r):
  
  path = join(p,name)
  Reducer  = Control.Reducer(method,M,r,path,Docs,DocsP)
  html = Control.HTMLBasisView(Reducer,path,createBasisViewer(join(path,"basisImages"),path,Reducer))
  html.generate()
  origImgP = join(path,"originalImages")
  repImgP = join(path,"representationImages")
  recImgP = join(path,"reconstructionImages")
  htmlPath = join(path,"objects")
  html = Control.HTMLObjectsView(Reducer,path,createOriginalViewer(origImgP,htmlPath,Reducer),createRepresentationViewer(repImgP,htmlPath,Reducer),createReconstructionViewer(recImgP,htmlPath,Reducer))
  html.generate()
  del Reducer
  del html

I = "/home/jecamargom/tmp/datasets/pain/"
M,Docs = Control.JPGtoMatrix(I,181,241)
DocsP = [join(I,f) for f in Docs]
print "[DEBUG] Max number in ORL faces matrix", M.max()
print "[DEBUG] Matrix dimensions : ", M.shape
p = "/home/jecamargom/tmp/experiments/pain2"
r = 14
generateFactorization("QLSA",Control.QLSA,M,p,r)
generateFactorization("QLSA2",Control.QLSA2,M,p,r)
generateFactorization("NMF",Control.NMF,M,p,r)
generateFactorization("VQ",Control.VQ,M,p,r)
generateFactorization("PCA",Control.PCA,M,p,r)

p = "/home/jecamargom/tmp/experiments/pain1"
r = 7
generateFactorization("QLSA",Control.QLSA,M,p,r)
generateFactorization("QLSA2",Control.QLSA2,M,p,r)
generateFactorization("NMF",Control.NMF,M,p,r)
generateFactorization("VQ",Control.VQ,M,p,r)
generateFactorization("PCA",Control.PCA,M,p,r)

p = "/home/jecamargom/tmp/experiments/pain3"
r = 4
generateFactorization("QLSA",Control.QLSA,M,p,r)
generateFactorization("QLSA2",Control.QLSA2,M,p,r)
generateFactorization("NMF",Control.NMF,M,p,r)
generateFactorization("VQ",Control.VQ,M,p,r)
generateFactorization("PCA",Control.PCA,M,p,r)
