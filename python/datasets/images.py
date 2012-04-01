import Image
import colorsys as cs
from os.path import join
import os
import numpy as np

def viewer(objects,path,prefix,dim,cellSize = 1):
    if cellSize == 1:
        for i in range(objects.shape[1]):
            toImage(objects[:,i],dim[0],path,prefix+str(i))
    else :
        for i in range(objects.shape[1]):
            b =  objects[:,i]
            h = dim[0]
            savePath = join(path,prefix+str(i)+".bmp")
            if not os.path.exists(savePath):
              im = scaledToArray(b,h,cellSize)
              x = Image.fromarray(im)
              x.save(savePath,"BMP")
            return savePath



def getColor(number,maxp,maxn,inverse):
    """Returns an array with three elments R, G and B with a certain level of black or red (depending on the sign of the numbre provided)"""
    if number >= 0:
      if inverse:
        ret = cs.hsv_to_rgb(0,0,abs(number/maxp))
      else:
        ret = cs.hsv_to_rgb(0,0,1-abs(number/maxp))
    else:
      if inverse:
        ret = cs.hsv_to_rgb(0,1-abs(number/maxn),1)
      else:
        ret = cs.hsv_to_rgb(0,abs(number/maxn),1)
    return [ret[0]*255.0,ret[1]*255.0,ret[2]*255.0]

def toArray(b,h):
    maxp = max(b)
    maxn = min(b)
    if maxn == 0:
        maxn = 0.1
    if maxp == 0:
        maxp = 0.1
    matrix = (b.reshape(h,-1))
    im = np.zeros((matrix.shape[0],matrix.shape[1],3),dtype=np.uint8)
    for i in range(im.shape[0]):
      for j in range(im.shape[1]):
        im[i,j]=getColor(matrix[i,j],maxp,maxn,True)
    return im

def toImage(b,h,path,name):
    """Stores in path an image with normalized colors from red to black according to the values in b with dimesions h x lenght_of_b/h"""
    savePath = join(path,name+".jpeg")
    if not os.path.exists(savePath):
      im = toArray(b,h)
      x = Image.fromarray(im)
      x.save(savePath,"JPEG")
    return savePath


def scaledToArray(b,h,cellSize):

    if (b.size%h)!=0:
      w = np.ceil(b.size/h)
      area = w*h
      b = np.append(b, [0]*(area-b.size))
      matrix = (b.reshape(h,w))
    else:
      matrix = (b.reshape(h,-1))
    #Now we scale the image array by multiplying it in the left by X and in the right by Y.
    r = matrix.shape[0]
    c = matrix.shape[1]
    X = np.zeros((r*cellSize,r))
    for i in range(r):
      X[:,i] = np.concatenate(((cellSize*i)*[0],[1]*cellSize,[0]*((r-i-1)*cellSize)))
    Y = np.zeros((c,c*cellSize))
    for j in range(c):
      Y[j,:] = np.concatenate(((cellSize*j)*[0],[1]*cellSize,[0]*((c-j-1)*cellSize)))
    matrix = np.dot(np.dot(X,matrix),Y)
    im = np.zeros((matrix.shape[0],matrix.shape[1],3),dtype=np.uint8)
    for i in range(im.shape[0]):
      for j in range(im.shape[1]):
        im[i,j]=getColor(matrix[i,j])
    return im


