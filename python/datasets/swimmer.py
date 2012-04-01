import scipy.io as sio
import numpy as np
import images

def loadMatrix(path='/home/saaperezru/datasets/swimmer/Y.mat',pattern='',amount=256):
    tmp = sio.loadmat(path)['Y']
    dimension = tmp.shape[0],tmp.shape[1]
    matrix = buildMatrix(tmp,amount)
    return matrix,dimension


def buildMatrix(M,i):
    ret = np.zeros((1024,i))
    for x in range(i):
        ret[:,x]=np.asarray(M[:,:,x]).reshape(-1)
    return ret

def viewer(objects,path,prefix,dim):
    images.viewer(objects,path,prefix,dim)
