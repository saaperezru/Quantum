class Basis:

  def __init__(self):
    self.feature = []

  def addFeature(self,weight,feature):
    self.feature.append((weight,feature))

  def orderFeatures(self,method):
    self.feature = method(self.feature)


class Object:
  def __init__(self,name,path):
    self.name = name
    self.path = path
    self.rep = []
    self.reconstruction = []
  def addRepresentation(self,weight,basis):
    """Adds a pair weight, basis to the representation of the object"""
    self.rep.append((weight,basis))
  def addReconstructionFeature(self,weight,feature):
    """Adds a pair weight, feature to the new reconstructed representation of the object"""
    self.reconstruction.append((weight,feature))
  def orderBasis(self,method):
    """Order the elements of the basis of the new space according to their weights using mehtod"""
    self.rep = method(self.rep)

class Feature:
  
  def __init__(self,name):
    self.name = name

class Reduction:
  
  def _simpleOrder(l):
    return l

  def __init__(self,X,r,features,documents,basisProcessor):
    self.orderFeauteresInBasis = self._simpleOrder 
    self.orderBasisInObject = self._simpleOrder 
    self.dimension = r
    self.data = X
    self.basis = []
    self.objects= []
    B,R,self.reconstructed = basisProcessor(X,r)
    self.basisM = B
    self.objectsM = R
    for j in range(r):
      tmp = Basis()
      for i in range(B.shape[0]):
        tmp.addFeature(B[i,j],features[i])
      self.basis.append(tmp)
    self.basis.orderFeatures(self.orderFeaturesInBasis)
    for j in range(R.shape[1]):
      for i in range(R.shape[0]):
        documents[j].addRepresentation(R[i,j],self.basis[i])
      for i in range(self.reconstructed.shape[0]):
        documents[j].addReconstructionFeauter(self.reconstructed[i,j],features[i])
      documents[j].orderBasis(self.orderBasisInObject)
  def getBasis(self):
    return orderBasisInObject
  def addBasis(self,basis):
    self.basis.append(basis)
  def addObject(self,obj):
    self.objects.append(obj)
