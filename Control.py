import numpy as np
import nmf
import scipy.cluster.vq as vq
import re
import os
import Image
import colorsys as cs
import PIL
from Modelo import Feature
from Modelo import Basis
from Modelo import Reduction
from Modelo import Object
from os.path import join,relpath
##############
#Factorizators
##############
def LSA(X,r):
  if r>X.shape[0]:
    print "[ERROR] Trying to extract more PCs than possible"
    return None,None,None
  U,S,V = np.linalg.svd(X)
  Uh = U[:,0:r]
  Sh = np.diag(np.array(S[0:r]))
  Vh = V[0:r,:]
#Letting (as in the paper) rep_{ij}=<psi_{j}|sigma_{i}> is saying rep = np.dot(Uh.T,X), which is the same as rep = np.dot(Sh,Vh) because Uh.T*X = Uh.T*U*S*V = Sh*Vh (Uh.T*U gives a very special matrix)
  rep = np.dot(Sh,Vh)
  basis = Uh
  #rep2 = np.dot(basis.T,X)
#  print "[DEBUG] Rep error = ", sum(rep-rep2)
  Xh = np.dot(basis,rep)
  return basis,rep,Xh
def PCA(X,r):
  #m = (1.0/Xo.shape[1])*np.dot(Xo,np.ones((Xo.shape[1],1)))
  m = np.asmatrix(np.mean(X,axis=1))
  print "[DEBUG] m shape : ", m.shape
  X = X - m.T 
  Y = (1.0/np.sqrt(X.shape[1]))*X.T
  U,S,V = np.linalg.svd(Y)
  del U
  del S
  del Y
  #Reduce the dimensionality of the  principal components space
  V = V[0:r,:]
  #In PCA we look for a matrix P such that Y = PX and Y has a covariance matrix with zeros in the off-diagonal terms and big numbers on the diagonal terms.
  #This P matrix is found by performing SVD over (1/sqrt(n))*X = U S VT and taking P = V. Here P is a transformation matrix from the canonical base space to the feature base space.
  # In this script we are interested on a matrix W such that X ~= WH, where W is a transformation matrix from the the feature basis space to the canonical basis space and H is the representation of the data in the feature space. So by linear algebra we know that W = P^-1, and so W = V^-1 = V^T. And H is obviously VX (come on V^T V X = X !!!!)
  rep = np.dot(V,X)
  return V.T,rep,np.dot(V.T,rep)

def QLSA(X,r):
  Xh = quantumNormalize(X)
  return LSA(Xh,r)
def QLSA2(X,r):
  Xn = quantumNormalize(X)
  B,R,Xh = LSA(Xn,r)
  del Xn
#We reconstruct X by projecting over the basis and normalizing (we normalize because we want to take advantage of the probability interpretation that pure states in the quantum document space has)
  Xh = quantumReconstruct(X,B)
  R = quantumRepresent(Xh,B)
#We square the basis to give it a probabilistic interpretation (as its a vector in the quantum documents space)
  B = np.multiply(B,B)
  return B,R,Xh
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
  C,label = vq.kmeans2(X.T,r,minit='points') #VQ K-means receives a matrix of observations x dimensions
  rep = np.zeros((r,X.shape[1]))
  for i in range(len(label)):
    rep[label[i],i]=1
  return C.T,rep,np.dot(C.T,rep)
   

####################
#quantumExtraMethods
####################

def quantumNormalize(X):
  Xn = X/np.dot(np.ones((X.shape[0],1)),np.dot(np.ones((1,X.shape[0])),X))
  print "[DEBUG] End first normalization step"
  Xn = np.sqrt(Xn)
  return Xn

def quantumReconstruct(X,B):
  #The projector over the space defined by the base B is Ps = BB^T
  Xh = np.dot(B.T,X)
  Xh = np.dot(B,Xh)
  #Now we need to normalize
  X2 = np.multiply(Xh,Xh)
  Xh = Xh / np.dot(np.ones((X.shape[0],1)),np.sqrt(np.dot(np.ones((1,X.shape[0])),X2)))
  del X2
  #We return the squared version so it has a probabilistic interpretation
  return np.multiply(Xh,Xh)
def quantumRepresent(X,B):
  #We want rep_{ij} to be <psi^hat_{i}|sigma_{k}>^2, which is equals to P(Z_{k}|d_{i}) : probability of latent topic Z_{k} given document i
# And we know that <psi^hat_{i}|sigma_{k}> = <sigma_{k}|psi^hat_{i}> because both vectors have real values and so rep = (Uh.T*X)^2
  R = np.dot(B.T,X)
  R = np.multiply(R,R)
  return R
############
#Reducers
############


class Reducer():
  def __init__(self,factorizator,X,r,path,D=None,P=None,F=None):
    self.factorizator = factorizator
    self.X = X
    self.R = r
    self.path  = path
    if F==None:
      F = [""]*X.shape[0]
    if D==None:
      D = []
      for i in range(X.shape[1]):
        D.append(str(i))
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
    reconstructionFile = join(path,"reconstruction.npy")
    #Verify if there is already a factorization in the path directory

    if(os.path.exists(basisFile) and os.path.exists(repFile) and os.path.exists(reconstructionFile)):
      #Load existing factorizationh
      print "[DEBUG] Loading existing factorization files"
      B = np.load(basisFile)
      R = np.load(repFile)
      Xh = np.load(reconstructionFile) 
    else:
      #Factorize and store results
      print "[DEBUG] Performing Factorization"
      B,R,Xh = factorizator(X,r)

      fileBasis = open(basisFile,'w')
      np.save(fileBasis,B)
      fileBasis.close()

      fileRep = open(repFile,'w')
      np.save(fileRep,R)
      fileRep.close()

      fileReconstruct = open(reconstructionFile,'w')
      np.save(fileReconstruct,Xh)
      fileReconstruct.close()
    #Finally build reduction
    print "[DEBUG] Basis :", B.shape," Documents: ",R.shape
    self.reduction = Reduction(X,r,self.features,self.documents,B,R,Xh) 

######################
#HTML Views Generators
######################

class HTMLBasisView():
  def __init__(self,reducer,path,basisViewGenerator=None):
    self.reducer = reducer
    self.path = path
    self.basisView= basisViewGenerator


  def generate(self):
    html = join(self.path,"basis.html")
    f = open(html,'w')
    f.write("<html>")
    j = 0
    for i in self.reducer.reduction.basis:
      f.write("<div><h3>B"+str(j)+"</h3>"+self.basisView.toHTML(i.features,"b"+str(j))+"</div>")
      j = j +1 
    f.write("</html>")
    f.close()
    return html


class HTMLObjectsView():
  def __init__(self,reducer,path,originalView,representationView,reconstructionView):
    self.originalView = originalView 
    self.representationView = representationView 
    self.reconstructionView= reconstructionView 
    self.path = path
    self.reducer = reducer
  def generate(self):
    path = self.path
    htmlFile = join(path,"objects.html")
    f = open(htmlFile,'w')
    f.write('<html><div style="width:100;overflow:scroll;height:700;float:left">')
    objectsDirectory = join(path,"objects")
    try:
      os.mkdir(objectsDirectory)
    except:
      print "Error Creating objects folder"
    for obj in self.reducer.reduction.objects:
      #print "[DEBUG] Generating HTML for object " + obj.name
      #Create one HTML file for each object
      objFilePath = join(objectsDirectory,obj.name+".html")
      objFile = open(objFilePath,'w')
      objFile.write("<html><div>")
      #In every HTMLFile there must be three sections:
      # 1. Original View
      objFile.write("<div>")
      objFile.write(self.originalView.toHTML(obj))
      objFile.write("</div>")
      # 2. Representation View
      objFile.write("<div>")
      objFile.write(self.representationView.toHTML(obj.rep,obj.name))
      objFile.write("</div>")
      # 3. Reconstruction View
      objFile.write("<div>")
      objFile.write(self.reconstructionView.toHTML(obj.reconstruction,obj.name))
      objFile.write("</div>")
      #Now end the file
      objFile.write("</div></html>")
      objFile.close()
      f.write("<a target='canvas' href='" + relpath(objFilePath,path) + "'>" + obj.name  + "</a><br>")
    f.write('</div><div style = "float:left"><iframe name="canvas" width=700 height=700></iframe></div></html>')

###########################
#TEXT Objects views generators
###########################
class textOriginalViewGenerator():
  def toHTML(self,obj):
    f = open(obj.path,'r')
    return "<div style='width=700;height=500'>"+f.read()+"</div>"
class textViewGenerator():
  def __init__(self,maxp,maxn):
    self.maxp = maxp
    self.maxn = maxn

  def toHTML(self,features,name):
    ret = "<table border='1'>"
    for f in features:
      ret = ret + "<tr><td bgcolor=" + self.getColor(f[0])  + ">" + f[1].name  + "</td><td>"+str(f[0])+"</td></tr>"
    ret = ret + "</table>"
    return ret
  

  def getColor(self,number):
    """Returns an HTML color with a certain level of black or red (depending on the sign of the numbre provided)"""
    if number >= 0:
      ret = cs.hsv_to_rgb(0,0,1-abs(number/self.maxp))
    else:
      ret = cs.hsv_to_rgb(0,abs(number/self.maxn),1)
    hexcolor = '#%02x%02x%02x' % (ret[0]*255,ret[1]*255,ret[2]*255)
    return hexcolor 

###########################
#IMAGE Objects views generators
###########################
class imageOriginalViewGenerator():
  """ A class for generating an HTML String for viewing the originaal version of a pgm image, i.e. transforming a PGM image into a JPG image and storing it in an accesible site for an HTML page"""
  def __init__(self,maxp,path,htmlPath,h,inverse=False,QLSA=False):
   self.generator = imageViewGenerator(maxp,-1,path,htmlPath,h,inverse,QLSA) 
 
  def toHTML(self,obj):
    #objArray = np.asarray(read_pgm(obj.path).reshape(-1))
    return "<img src='"+relpath(self.generator.toImage(obj.data,obj.name),self.generator.htmlPath)+"'></img>"

class imageViewGenerator():

  def __init__(self,maxp,maxn,path,htmlPath,h,inverse=False,QLSA=False):
    self.maxp = float(maxp)
    self.maxn = maxn
    self.path = path
    self.h = h
    self.htmlPath = htmlPath
    self.inverse = inverse
    self.QLSA = QLSA

  def getColor(self,number):
    """Returns an array with three elments R, G and B with a certain level of black or red (depending on the sign of the numbre provided)"""
    if number >= 0:
      if self.inverse:
        ret = cs.hsv_to_rgb(0,0,abs(number/self.maxp))
      else:
        ret = cs.hsv_to_rgb(0,0,1-abs(number/self.maxp))
    else:
      if self.inverse:
        ret = cs.hsv_to_rgb(0,1-abs(number/self.maxn),1)
      else:
        ret = cs.hsv_to_rgb(0,abs(number/self.maxn),1)
    return [ret[0]*255.0,ret[1]*255.0,ret[2]*255.0]

  def toArray(self,b,h):
    #print "[DEBUG] Siize:" + str(b.size) + " - "  +str(h)
    #print "[DEBUG] imageViewGenerator : b.size = ",b.size," H = ",h," W = " , b.size/h
    matrix = (b.reshape(h,-1))
    im = np.zeros((matrix.shape[0],matrix.shape[1],3),dtype=np.uint8)
    for i in range(im.shape[0]):
      for j in range(im.shape[1]):
        im[i,j]=self.getColor(matrix[i,j])
    return im

  def toImage(self,b,name):
    """Stores in path an image with normalized colors from red to black according to the values in b with dimesions h x lenght_of_b/h"""
    savePath = join(self.path,name+".jpeg")
    if not os.path.exists(savePath):
      h = self.h
      im = self.toArray(b,h)
      x = Image.fromarray(im)
      x.save(savePath,"JPEG")
      del im
      del x
    return savePath
  def toHTML(self,featuresList,name):
    b = []
    for i in featuresList:
      b.append(i[0])
    if QLSA: 
      self.maxp = max(b)
    return "<img src='"+relpath(self.toImage(np.array(b),name),self.htmlPath)+"'></img>"

class imageScaledViewGenerator(imageViewGenerator):
  def __init__(self,maxp,maxn,path,htmlPath,h,cellSize):
    imageViewGenerator.__init__(self,maxp,maxn,path,htmlPath,h)
    self.cellSize = cellSize
  def toArray(self,b,h):
    if (b.size%h)!=0:
      w = np.ceil(b.size/h)
      area = w*h
      b = np.append(b, [0]*(area-b.size))
      matrix = (b.reshape(h,w))
    else:
      matrix = (b.reshape(h,-1))
    #print "[DEBUG] imageScaledViewGenerator : b.size = ",b.size," H = ",h," W = " , b.size/h
    #Now we scale the image array by multiplying it in the left by X and in the right by Y.
    r = matrix.shape[0]
    c = matrix.shape[1]
    X = np.zeros((r*self.cellSize,r))
    for i in range(r):
      X[:,i] = np.concatenate(((self.cellSize*i)*[0],[1]*self.cellSize,[0]*((r-i-1)*self.cellSize)))
    Y = np.zeros((c,c*self.cellSize))
    for j in range(c):
      Y[j,:] = np.concatenate(((self.cellSize*j)*[0],[1]*self.cellSize,[0]*((c-j-1)*self.cellSize)))
    matrix = np.dot(np.dot(X,matrix),Y)
    im = np.zeros((matrix.shape[0],matrix.shape[1],3),dtype=np.uint8)
    for i in range(im.shape[0]):
      for j in range(im.shape[1]):
        im[i,j]=self.getColor(matrix[i,j])
    return im


#########################################
# Methods for getting matrix of images
########################################

def imagesMatrix(path,imageSize = 10304,byteorder = '>'):
  """Returns a matrix formed with the pgm images in path along with a list with the names of the image fiels in order of apparence in the matrix"""
  listing = os.listdir(path)
  listing.sort()
  count = 0
  docFiles = []
  for infile in listing:
    count = count + 1
    docFiles.append(infile)
  matrix = np.zeros((imageSize,count))
  for i in range(len(listing)):
    matrix[:,i]=np.asarray(read_pgm(join(path,listing[i]),byteorder)).reshape(-1)
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

def JPGtoArray(path):
  image = PIL.Image.open(path)
  im = np.asarray(image)
  ret = np.empty((1,im.shape[0]*im.shape[1]))
  k = 0
  for i in range(im.shape[0]):
    for j in range(im.shape[1]):
      ret[0,k] = cs.rgb_to_hsv(im[i,j,0],im[i,j,1],im[i,j,2])[2]
      k = k+1
  return ret

def JPGtoMatrix(path,w,h):
  """Returns a matrix formed with the pgm images in path along with a list with the names of the image fiels in order of apparence in the matrix"""
  listing = os.listdir(path)
  listing.sort()
  count = 0
  docFiles = []
  for infile in listing:
    count = count + 1
    docFiles.append(infile)
  matrix = np.zeros((w*h,count))
  for i in range(len(listing)):
    matrix[:,i]=JPGtoArray(join(path,listing[i]))
  return matrix,listing

#################################
# Methods for getting text matrix
#################################

class textMatrixGenerator():

  def __init__(self, stopwords, ignorechars, path):
    self.stopwords = stopwords 
    self.ignorechars = ignorechars 
    self.wdict = {} 
    self.dcount = 0
    self.files = []
    self.words = []
    listing = os.listdir(path)
    listing.sort()
    for count in range(len(listing)):
      f = open(join(path,listing[count]))
      self.parse(f.read())
      self.files.append(listing[count])
  
  
  def parse(self, doc):
    words = doc.split(); 
    for w in words:
      w = w.lower().translate(None, self.ignorechars) 
      if w in self.stopwords:
        continue
      elif w in self.wdict:
        self.wdict[w].append(self.dcount)
      else:
        self.wdict[w] = [self.dcount]
    self.dcount += 1
  
  def build(self):
    self.keys = [k for k in self.wdict.keys() if len(self.wdict[k]) > 1] 
    self.keys.sort() 
    self.A = np.zeros([len(self.keys), self.dcount]) 
    for i, k in enumerate(self.keys):
      self.words.append(k)
      for d in self.wdict[k]:
        self.A[i,d] += 1
    return self.A
