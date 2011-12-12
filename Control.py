import numpy as np
import pymf as nmf
import scipy.cluster.vq as vq
import re
import os
import Image
import colorsys as cs
from Modelo import Feature
from Modelo import Basis
from Modelo import Reduction
from Modelo import Object
from os.path import join

##############
#Factorizators
##############
def PCA(X,r):
  if r>X.shape[0]:
    print "[ERROR] Trying to extract more PCs than possible"
    return None,None,None
  U,S,V = np.linalg.svd(X)
  Uh = U[:,0:r]
  Sh = np.diag(np.array(S[0:r]))
  Vh = V[0:r,:]
  rep = np.dot(Sh,Vh)
  basis = Uh
  Xh = np.dot(basis,rep)
  return basis,rep,Xh

def QLSA(X,r):
  Xh = quantumNormalize(X)
  return PCA(Xh,r)

def NMF(X,r):
  """Receives a numpy matrix X and returns the basis, the representation and the reconstructed matrix from performing NMF
    
    Parameters:
    ----------
      - X : numpy matrix
          The data matrix with dimensions num_features x num_samples
      - r : int
          The desired dimension of the new space

    Returns:
    -------
      - basis : numpy matrix
          A matrix with each of the basis vector for the new space in its columns.
      - rep : numpy matrix
          A matrix in which each column corresponds to a document represented in the new space. 
      - Xh : numpy matrix
          The reconstructed version of the original version using the new space, i.e. Xh = <basis,rep>, where the frobenius norm of the difference |X-Xh| is minimum and the basis and the representation are restricted to be non negative.

    Example:
    -------
   
  """
  nmf_mdl = nmf.NMF(X,num_bases=r)
  nmf_mdl.initialization()
  nmf_mdl.factorize()
  return nmf_mdl.W,nmf_mdl.H,np.dot(nmf_mdl.W,nmf_mdl.H)

def VQ(X,r):
  C,label = vq.kmeans2(X.T,r) #VQ K-means receives a matrix of observations x dimensions
  rep = zeros((r,X.shape[1]))
  for i in range(len(label)):
    rep[label[i],i]=1
  return C.T,rep,np.dot(C.T,rep)
   
def VQ2(X,r):
  kmeans_mdl = nmf.Kmeans(X,num_bases=r)
  kmeans_mdl.factorize()
  return kmeans_mdl.W,kmeans_mdl.H,np.dot(kmeans_mdl.W,kmeans_mdl.H)

####################
#quantumExtraMethods
####################

def quantumNormalize(X):
  Xn = X/np.dot(np.ones((X.shape[0],1)),np.dot(np.ones((1,X.shape[0])),X))
  print "[DEBUG] End first normalization step"
  Xn = np.sqrt(Xn)
  return Xn

def quantumReconstruct(X,B):
  Xh = np.dot(B.T,X)
  Xh = np.dot(B,Xh)
  X2 = np.multiply(X,X)
  Xh = Xh / np.dot(np.ones((X.shape[0],1)),np.sqrt(np.dot(np.ones((1,X.shape[0])),X2)))
  return Xh

############
#Reducers
############


class Reducer():
  def __init__(factorizator,X,r,path,D=None,P=None,F=None):
    self.factorizator = factorizator
    self.X = X
    self.R = r
    self.path  = path
    if F==None:
      F = [""]*X.shape[0]
    if D==None:
      D = [""]*X.shape[1]
    if P == None :
      P = [""]*X.shape[1]
    self.features = []
    self.documents = []
    for i in F:
      self.features.append(Feature(i))
    for i in range(len(D)):
      self.documents.append(Object(D[i],P[i])) 
    #Define directories for storing or loading factorization results
    basisFile = join(path,"basis.npy")
    repFile = join(path,"rep.npy")
    reconstructionFile = 
    #Verify if there is already a factorization in the path directory

    if(os.path.exists(basisFile) and os.path.exists(repFile) and os.path.exists(reconstructionFile)):
      #Load existing factorization
      B = np.load(basisFile)
      R = np.load(repFile)
      Xh = np.load(reconstructionFile) 
    else:
      #Factorize and store results
      B,R,Xh = factorizator(X,r)

      fileBasis = open(basisFile,'w')
      np.save(fileBasis,B)
      fileBasis.close()

      fileRep = open(repFile,'w')
      np.save(fileRep,R)
      fileRep.close()

      fileReconstruct = open(join(path,"reconstruct.npy"),'w')
      np.save(fileReconstruct,Xh)
      fileReconstruct.close()
    #Finally build reduction
    self.reduction = Reduction(X,r,self.features,self.documents,B,R,Xh) 

######################
#HTML Views Generators
######################

class HTMLBasisView():
  def __init__(self,reducer,path,basisViewGenerator=None):
    self.reducer = reducer
    self.path = path
    if basisViewGenerator == None:
      basisDirectory = join(path,"basisImages")
      try:
        os.mkdir(basisDirectory)
        self.basisView =  basisToImage(self.reduction.basisM.max(),self.reduction.basisM.min(),112,basisDirectory)
    else:
      self.basisView= basisViewGenerator


  def generate(self):
    html = join(self.path,"basis.html")
    f = open(html,'w')
    f.write("<html>")
    j = 0
    for i in self.reducer.reduction.basis:
      f.write(self.basisView.toHTML(i.feature,"b"+str(j))
    f.write("</html>")
    f.close()
    return html


class HTMLObjectsView():

  def setOriginalObjectViewGenerator(self,generator):
    self.originalView = generator
  def RepresentationViewGenerator(self,generator):
    self.representationView = generator
  def ReconstructedViewGenerator(self,generator):
    self.reconstructedView = generator 
    self.originalView = None
    self.representationView = None
    self.reconstructedView = None

###########################
#Objects views generators
###########################

class imageOriginalViewGenerator():
  def toHTML(self,obj):
    return "<img src='"+obj.path+"'></img>"

class imageViewGenerator():

  def __init__(self,maxp,maxn,path,h):
    self.maxp = maxp
    self.maxn = maxn
    self.path = path
    self.h = self.h

  def getColor(self,number):
    """Returns an array with three elments R, G and B with a certain level of black or red (depending on the sign of the numbre provided)"""
    if number >= 0:
      ret = cs.hsv_to_rgb(0,0,1-abs(number/self.maxp))
    else:
      ret = cs.hsv_to_rgb(0,abs(number/self.maxn),1)
    return [ret[0]*255,ret[1]*255,ret[2]*255]

  def toArray(self,b,h):
    matrix = (b.reshape(h,-1))
    im = np.zeros((matrix.shape[0],matrix.shape[1],3),dtype=np.uint8)
    for i in range(im.shape[0]):
      for j in range(im.shape[1]):
        im[i,j]=self.getColor(matrix[i,j])
    return im

  def toImage(self,b,name):
    """Stores in path an image with normalized colors from red to black according to the values in b with dimesions h x lenght_of_b/h"""
    im = self.toArray(b,h)
    x = Image.fromarray(im)
    savePath = join(self.path,name+".jpeg")
    x.save(savePath,"JPEG")
    return savePath
  def toHTML(self,featuresList,name):
    b = []
    for i in featuresList:
      b.append(i[0])
    return "<img src='"+toImage(b,self.h,name)+"'></img>"

    

#########################################
# Methods for getting matrix of images
########################################

def imagesMatrix(path):
  """Returns a matrix formed with the pgm images in path along with a list with the names of the image fiels in order of apparence in the matrix"""
  listing = os.listdir(path)
  listing.sort()
  count = 0
  docFiles = []
  for infile in listing:
    count = count + 1
    docFiles.append(infile)
  matrix = np.zeros((10304,count))
  for i in range(len(listing)):
    matrix[:,i]=np.asarray(read_pgm(join(path,listing[i]))).reshape(-1)
  return matrix,listing

def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html
    
    Reference : http://stackoverflow.com/questions/7368739/numpy-and-16-bit-pgm
    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,dtype='u1' if int(maxval) < 256 else byteorder+'u2',count=int(width)*int(height),offset=len(header)).reshape((int(height), int(width)))
