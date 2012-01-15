class Basis:

  def __init__(self):
    self.features = []

  def addFeature(self,weight,feature):
    self.features.append((weight,feature))

  def orderFeatures(self,method):
    self.features = method(self.features)


class Object:
  def __init__(self,name,path):
    self.name = name
    self.path = path
    self.rep = []
    self.reconstruction = []
    self.data = None
  def addRepresentation(self,weight,basis):
    """Adds a pair weight, basis to the representation of the object"""
    self.rep.append((weight,basis))
  def addReconstructionFeature(self,weight,feature):
    """Adds a pair weight, feature to the new reconstructed representation of the object"""
    self.reconstruction.append((weight,feature))
  def orderBasis(self,method):
    """Order the elements of the basis of the new space according to their weights using mehtod"""
    self.rep = method(self.rep)
  def setData(self,data):
    self.data=data

class Feature:
  
  def __init__(self,name):
    self.name = name

class Reduction:
  
  def _simpleOrder(self,l):
    return l

  def __init__(self,X,r,features,documents,B,R,Xh):
    self.orderFeaturesInBasis= self._simpleOrder 
    self.orderBasisInObject = self._simpleOrder 
    self.dimension = r
    self.data = X
    self.basis = []
    self.objects= documents
    self.reconstructed = Xh
    self.basisM = B
    self.objectsM = R
    for j in range(self.dimension):
      tmp = Basis()
      for i in range(self.data.shape[0]):
        tmp.addFeature(self.basisM[i,j],features[i])
      self.basis.append(tmp)
    for j in range(self.data.shape[1]):
      for i in range(self.dimension):
        documents[j].addRepresentation(self.objectsM[i,j],self.basis[i])
      for i in range(self.data.shape[0]):
        documents[j].addReconstructionFeature(self.reconstructed[i,j],features[i])
      documents[j].setData(X[:,j])
  def getBasis(self):
    return orderBasisInObject
  def addBasis(self,basis):
    self.basis.append(basis)
  def addObject(self,obj):
    self.objects.append(obj)
