import Control

I = '/home/tuareg/Documents/UNAL/7mo semestre/Machine Learning 2/TMP/datasets/ORLFull'

M,L = Control.imagesMatrix(I)
M = Control.quantumNormalize(M)
quantumImagesDirectory = '/home/tuareg/Documents/UNAL/7mo semestre/Machine Learning 2/TMP/datasets/ORLFullQuantumNormalized/'
imageGenerator = Control.imageViewGenerator(M.max(),M.min(),quantumImagesDirectory,quantumImagesDirectory,112,True)

for i in range(M.shape[1]):

  imageGenerator.toImage(M[:,i],L[i])
