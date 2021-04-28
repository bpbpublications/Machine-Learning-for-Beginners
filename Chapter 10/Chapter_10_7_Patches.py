from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image
from matplotlib import pyplot as plt
import numpy as np
from skimage.color import rgb2gray

img1=load_sample_image('flower.jpg')
img1=rgb2gray(img1)
plt.imshow(img1)
plt.show()
patches=image.extract_patches_2d(img1,(2,2))
plt.imshow(patches[10000,:,:])
plt.show()