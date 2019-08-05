# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 09:54:30 2018

@author: Nasif
"""

import numpy as np 
import matplotlib.pyplot as plt
from struct import unpack
from sklearn.externals import joblib
from scipy import misc
from PIL import Image
import PIL.ImageOps

from sklearn.linear_model import LogisticRegression
import pandas
from sklearn import model_selection
import pickle
# Used for Confusion Matrix
from sklearn import metrics
import seaborn as sns

def loadmnist(imagefile, labelfile):

    # Open the images with gzip in read binary mode
    images = open(imagefile, 'rb')
    labels = open(labelfile, 'rb')

    # Get metadata for images
    images.read(4)  # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    # Get metadata for labels
    labels.read(4)
    N = labels.read(4)
    N = unpack('>I', N)[0]

    # Get data
    x = np.zeros((N, rows*cols), dtype=np.uint8)  # Initialize numpy array
    y = np.zeros(N, dtype=np.uint8)  # Initialize numpy array
    for i in range(N):
        for j in range(rows*cols):
            tmp_pixel = images.read(1)  # Just a single byte
            tmp_pixel = unpack('>B', tmp_pixel)[0]
            x[i][j] = tmp_pixel
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]

    images.close()
    labels.close()
    return (x, y)

#train_img, train_lbl = loadmnist('data/train-images-idx3-ubyte'
#                                 , 'data/train-labels-idx1-ubyte')
test_img, test_lbl = loadmnist('data/t10k-images-idx3-ubyte'
                               , 'data/t10k-labels-idx1-ubyte')
#print(test_img[0])

logisticRegr = joblib.load('traindata.pkl')
#plt.figure(figsize=(20,4))
#for index, (image, label) in enumerate(zip(test_img[0:1], test_lbl[0:1])):
#    plt.subplot(1, 5, index + 1)
#    plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
#    plt.title('Test: %i\n' % label, fontsize = 20)
print(logisticRegr.predict(test_img[0].reshape(1,-1)))
print(test_img[0].dtype)
#print(test_img[0])

img = misc.imread("cur2.jpg") 
img = misc.imresize(img, (28, 28))
print(img)
print(img.dtype)

#x_test = []
#
#for eachRow in img:
#	for eachPixel in eachRow:
#		x_test.append(sum(eachPixel)/3.0)
#print(logisticRegr.predict([x_test]))

x=input("enter any key")