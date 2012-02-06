import nibabel as nib
import Control

img = nib.load("../../../datasets/oasis/OAS1_0227_MR1_mpr_n4_anon_111_t88_masked_gfc.hdr").get_data()[:,:,:,0]
imagesDirectory = "./"
height =176 
index = 100
im = img[:,100,:]
imgViewer = Control.imageViewGenerator(im.max(),0,imagesDirectory,imagesDirectory,height,False)
imgViewer.toImage(im.reshape(1,-1),"f"+str(index))
