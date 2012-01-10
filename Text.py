from os.path import join
import os
import Control
def createRepresentationViewer(imagesDirectory,htmlPath,Reducer):
  try:
    os.mkdir(imagesDirectory)
  except:
    print "Error Creating folder",imagesDirectory
  return  Control.imageScaledViewGenerator(Reducer.reduction.objectsM.max(),Reducer.reduction.objectsM.min(),imagesDirectory,htmlPath,1,10)
def createBasisViewer(Reducer):
  return Control.textViewGenerator(Reducer.reduction.basisM.max(),Reducer.reduction.basisM.min())
def createReconstructionViewer(Reducer):
  return Control.textViewGenerator(Reducer.reduction.objectsM.max(),Reducer.reduction.objectsM.min())
def createOriginalViewer():
  return Control.textOriginalViewGenerator()


def generateFactorization(name,method,M,Docs,DocsP,F):
  p = "/home/jecamargom/tmp/experiments/text1/"
  path = join(p,name)
  Reducer  = Control.Reducer(method,M,r,path,Docs,DocsP,F)
  #Basis View Generation
  html = Control.HTMLBasisView(Reducer,path,createBasisViewer(Reducer))
  html.generate()
  repImgP = join(path,"representationImages")
  htmlPath = join(path,"objects")
  html = Control.HTMLObjectsView(Reducer,path,createOriginalViewer(),createRepresentationViewer(repImgP,htmlPath,Reducer),createReconstructionViewer(Reducer))
  html.generate()
  del Reducer
  del html

#Path for storing resutls from factorization into HTML form and factorization results
I = "/home/jecamargom/tmp/datasets/text1/"

r = 5

#Construction of term-document matrix
ignorechars = ''':,'''
stopwords = ["a","of","lab","abc","for","the","applications","machine","management","opinion","testing","relation","error","measurement","perceived","iv","paths","widths","intersection","in","unordered","binary","random","and","engineering"]
generator = Control.textMatrixGenerator(stopwords,ignorechars,I)
M = generator.build()
#Construction of names and paths arrays
Docs = generator.files
DocsP = [join(I,f) for f in Docs]
#Construction of features arrayy
F = generator.words 
#Reduction
print "[DEBUG]"
print M.shape

generateFactorization("QLSA",Control.QLSA,M,Docs,DocsP,F)
generateFactorization("QLSA2",Control.QLSA2,M,Docs,DocsP,F)
generateFactorization("NMF",Control.NMF,M,Docs,DocsP,F)
generateFactorization("VQ",Control.VQ,M,Docs,DocsP,F)
generateFactorization("PCA",Control.PCA,M,Docs,DocsP,F)

