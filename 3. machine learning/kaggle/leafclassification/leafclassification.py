# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 14:49:40 2017

https://www.kaggle.com/lorinc/leaf-classification/feature-extraction-from-images/notebook

@author: FloyerXie
"""

import numpy as np

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import measure
import scipy.ndimage as ndi

# matplotlib setup
# matplotlib inline
from pylab import rcParams

img = mpimg.imread("images/53.jpg")
cy, cx = ndi.center_of_mass(img)

plt.imshow(img, cmap="Set2")
plt.scatter(cx, cy)
plt.show()

contours = measure.find_contours(img, .8)
contour = max(contours, key=len)

plt.plot(contour[::,1], contour[::, 0], linewidth=0.5)
plt.imshow(img, cmap="Set3")
plt.show()


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return [rho, phi]

# just calling the transformation on all pairs in the set
polar_contour = np.array([cart2pol(x, y) for x, y in contour])

# and plotting the result
plt.plot(polar_contour[::,1], polar_contour[::,0], linewidth=0.5)
plt.show()