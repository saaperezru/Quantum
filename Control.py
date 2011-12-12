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

##################################

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

###################################

def ReduceQLSA(X,r,path,F=None,D=None,P=None):
  return Reduce(QLSA,X,r,path,F,D,P)

def ReduceNMF(X,r,path,F=None,D=None,P=None):
  return Reduce(NMF,X,r,path,F,D,P)

def ReducePCA(X,r,path,F=None,D=None,P=None):
  return Reduce(PCA,X,r,path,F,D,P)

def ReduceVQ(X,r,path,F=None,D=None,P=None):
  return Reduce(VQ,X,r,path,F,D,P)

def Reduce(M,X,r,path,F=None,D=None,P=None):
  if F==None:
    F = [""]*X.shape[0]
  if D==None:
    D = [""]*X.shape[1]
    P = [""]*X.shape[1]
  features = []
  documents = []
  for i in F:
    features.append(Feature(i))
  for i in range(len(D)):
    documents.append(Object(D[i],P[i])) 
  red = Reduction(X,r,features,documents,M) 
  im  = basisToImage(red.basisM.max(),red.basisM.min())
  for i in range(r):
    im.toImage(red.basisM[:,i],112,join(path,"b"+str(i)))
  return red,documents,im

########################################

def imagesMatrix(path):
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
  return matrix

class basisToImage():

  def __init__(self,maxp,maxn):
    self.maxp = maxp
    self.maxn = maxn

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

  def toImage(self,b,h,path):
    """Stores in path an image with normalized colors from red to black according to the values in b with dimesions h x lenght_of_b/h"""
    im = self.toArray(b,h)
    x = Image.fromarray(im)
    x.save(path,"JPEG")

  

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
