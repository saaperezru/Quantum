import numpy as np
import nmf
import scipy.cluster.vq as vq



def LSA(X,r,path):
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
def PCA(X,r,path):
  print "[DEBUG] Executing PCA"
  #m = (1.0/Xo.shape[1])*np.dot(Xo,np.ones((Xo.shape[1],1)))
  m = np.asmatrix(np.mean(X,axis=1))
  print "[DEBUG] m shape : ", m.shape
  Xnormal = X - m.T 
  #Y = (1.0/np.sqrt(X.shape[1]))*Xnormal
  U,S,V = np.linalg.svd(Xnormal)
  #Reduce the dimensionality of the  principal components space
  U = U[:,0:r]
  #In PCA we look for a matrix P such that Y = PX and Y has a covariance matrix with zeros in the off-diagonal terms and big numbers on the diagonal terms.
  #This P matrix is found by performing SVD over (1/sqrt(n))*X = U S VT and taking P = V. Here P is a transformation matrix from the canonical base space to the feature base space.
  # In this script we are interested on a matrix W such that X ~= WH, where W is a transformation matrix from the the feature basis space to the canonical basis space and H is the representation of the data in the feature space. So by linear algebra we know that W = P^-1, and so W = V^-1 = V^T. And H is obviously VX (come on V^T V X = X !!!!)
  rep = np.dot(U.T,Xnormal)
  rec = np.dot(U,rep) + m.T 
  #np.save('/home/saaperezru/QLSA/X.npy',X)
  return U,rep,rec

def QLSA(X,r,path):
  Xh = quantumNormalize(X)
  return LSA(Xh,r)
def QLSA2(X,r,path):
  Xn = quantumNormalize(X)
  B,R,Xh = LSA(Xn,r,path)
  del Xn
#We reconstruct X by projecting over the basis and normalizing (we normalize because we want to take advantage of the probability interpretation that pure states in the quantum document space has)
  Xh = quantumReconstruct(X,B)
  R = quantumRepresent(Xh,B)
#We square the basis to give it a probabilistic interpretation (as its a vector in the quantum documents space)
  B = np.multiply(B,B)
  Xh = np.multiply(Xh,Xh)
  return B,R,Xh
def NMF(X,r,path):
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

def VQ(X,r,path):
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
  return Xh
def quantumRepresent(X,B):
  #We want rep_{ij} to be <psi^hat_{i}|sigma_{k}>^2, which is equals to P(Z_{k}|d_{i}) : probability of latent topic Z_{k} given document i
# And we know that <psi^hat_{i}|sigma_{k}> = <sigma_{k}|psi^hat_{i}> because both vectors have real values and so rep = (Uh.T*X)^2
  R = np.dot(B.T,X)
  R = np.multiply(R,R)
  return R

