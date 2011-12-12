class Basis:

  def __init__(self):
    self.feature = []

  def addFeature(self,weight,feature):
    self.feature.append((weight,feature))

class Object:
  def __init__(self,name,path):
    self.name = name
    self.path = path
    self.rep = []
  def addRepresentation(self,weight,basis):
    self.rep.append((weight,basis))

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
    for j in range(R.shape[1]):
      for i in range(R.shape[0]):
        documents[j].addRepresentation(R[i,j],self.basis[i])


  def addBasis(self,basis):
    self.basis.append(basis)
  def addObject(self,obj):
    self.objects.append(obj)
