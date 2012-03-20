import scipy.io as sio
from os.path import join
from os import mkdir,remove
import numpy as np

def divide(originalList,positiveList,negativeList):

    positiveSet = set([])
    negativeSet = set([])
    
    positiveFile = open(positiveList,'r')
    negativeFile = open(negativeList,'r')

    for record in positiveFile:
        positiveSet.add(record.split('.')[-2].strip())

    for record in negativeFile:
        negativeSet.add(record.split('.')[-2].strip())


    originalFile = open(originalList,'r')
    positiveIndex = []
    negativeIndex  = []
    i = 0
    for record in originalFile:
        i = i+1
        if record.split('.')[0].strip() in positiveSet:
            positiveIndex.append(i)
            continue
        if record.split('.')[0].strip() in negativeSet:
            negativeIndex.append(i)
            continue
    return (positiveIndex,negativeIndex)

def translate(indexList,matrix,path,clas):
    svmFile = open(path,'a')
    for i in indexList:
        j = 1
        svmFile.write(clas + " ")
        for v in matrix[:,i-1]:
          svmFile.write(str(j) + ':' + str(v) + ' ')  
          j = j+1
        svmFile.write('\n')
        
    svmFile.close()

def buildR(matrix,p,path):
    rep = np.zeros((matrix.shape[0],len(p)))
    for i in range(len(p)):
        rep[:,i] = matrix[:,p[i]-1]
    sio.savemat(path,{'rep':rep})

for r in [80,40,20,10,5]:
#for r in [80]:
    matrixPathBasis = join('/home/saaperezru/QLSA/experiments/oasis_group1/','full'+str(r))
    listBasis = join('/home/saaperezru/QLSA/experiments7CCC/oasis_group1/')
    svmData = join('/home/saaperezru/SVM/experiments/oasis_group1/','full'+str(r))
    rPath = join('/home/saaperezru/QLSA/experiments7CCC/oasis_group1','full'+str(r))

    try:
        mkdir(svmData)
        mkdir(rPath)
    except:
        print "[DEBUG] SVM Data dir already exists"

    matrixPath = join(matrixPathBasis,'R2.mat')
    oList = join(listBasis,'originalList.txt')
    pList = join(listBasis,'positiveList.txt')
    nList = join(listBasis,'negativeList.txt')


    R = sio.loadmat(matrixPath)
    (p,n) = divide(oList,pList,nList)

    svmF = join(svmData,'data.txt')
    remove(svmF)
    translate(p,R['rep'],svmF,"1")
    translate(n,R['rep'],svmF,"0")

    positiveR = join(rPath,'positiveR.mat')
    negativeR = join(rPath,'negativeR.mat')
    buildR(R['rep'],p,positiveR)
    buildR(R['rep'],n,negativeR)

