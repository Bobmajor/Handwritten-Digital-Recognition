# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 09:18:56 2018

@author: Nasif
"""

#matplotlib inline
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import pandas
from sklearn import model_selection
from sklearn.externals import joblib
from sklearn import ensemble
import pickle
from sklearn.svm import SVC

# Used for Confusion Matrix
from sklearn import metrics
import seaborn as sns


# Used for Loading MNIST
from struct import unpack
# Change data_home to wherever to where you want to download your data

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

train_img, train_lbl = loadmnist('data/train-images-idx3-ubyte'
                                 , 'data/train-labels-idx1-ubyte')
test_img, test_lbl = loadmnist('data/t10k-images-idx3-ubyte'
                               , 'data/t10k-labels-idx1-ubyte')

print(train_img.shape)

#plt.figure(figsize=(20,4))
#for index, (image, label) in enumerate(zip(train_img[0:5], train_lbl[0:5])):
#    plt.subplot(1, 5, index + 1)
#    plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
#    plt.title('Training: %i\n' % label, fontsize = 20)


#logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr = SVC(gamma = 0.1)
logisticRegr.fit(train_img, train_lbl)

joblib.dump(logisticRegr, 'traindata.pkl')




logisticRegr.predict(test_img[0].reshape(1,-1))
score = logisticRegr.score(test_img, test_lbl)
print(score)
