from matplotlib import pyplot as plt
import numpy as np
import math
from scipy import misc
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.datasets import load_sample_images

def rgb2gray(rgb):                                    #converting rgb into gray scale
    r,g,b=rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]
    gray=0.2989*r+0.5870*g+0.1140*b
    return gray


data = load_sample_images()     
len(data.images)                
img1 = data.images[1]
img1.shape              
plt.imshow(img1)

img2=rgb2gray(img1)
plt.imshow(img2)
fd, hog_image = hog(img1, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualize=True, multichannel=True)
plt.imshow(hog_image)