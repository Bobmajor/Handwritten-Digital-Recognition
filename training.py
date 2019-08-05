# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 08:38:40 2018

@author: Nasif
"""

from sklearn import datasets
from sklearn.svm import SVC
from scipy import misc
from PIL import Image
import PIL.ImageOps


digits = datasets.load_digits()
features = digits.data 
labels = digits.target

clf = SVC(gamma = 0.001)
clf.fit(features, labels)

image = Image.open('cur.jpg')
if image.mode == 'RGBA':
    r,g,b,a = image.split()
    rgb_image = Image.merge('RGB', (r,g,b))

    inverted_image = PIL.ImageOps.invert(rgb_image)

    r2,g2,b2 = inverted_image.split()

    final_transparent_image = Image.merge('RGBA', (r2,g2,b2,a))

    final_transparent_image.save('cur1.png')

else:
    inverted_image = PIL.ImageOps.invert(image)
    inverted_image.save('cur1.png')
    
img = misc.imread("cur1.png") 
img = misc.imresize(img, (8,8))
img = img.astype(digits.images.dtype)
img = misc.bytescale(img, high=16, low=0)


x_test = []

for eachRow in img:
	for eachPixel in eachRow:
		x_test.append(sum(eachPixel)/3.0)


print(clf.predict([x_test]))