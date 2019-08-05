# digital_handwriting_recognition
A python project that prompts a paint window where one can draw digital image of English digits and it can recognize the corresponding digit.

# Requirements
This code was run on python 3.6 and requires the latest versions of the followings:

sklearn
PIL
matplotlib
numpy
pdf2image

# File Usage
training.py, test.py and training.py are individual code segments that were used to test various features of the final code.

training_final.py is the code for training the dataset. Here the MNIST dataset was used (It was tested on digits dataset as well). 

The Data folder holds necessary data files to run training_final.py

The image folder holds some digit images that was used to test different features via training.py, test.py and predict.py

project_demo.py is the final code that when run as a python program, it prompts a window that allows user to draw a digital digit on the window and then the code tries to recognize/predict which digit it is.

# Result
The code reaches approximately 80% accuracy. The shortcoming of accuracy occurs because:
1. It was trained on MNIST dataset which asks for more training images.
2. A systemic loss of information because of the steps:
    a. taking a snap of the paint window as a pdf,
    b. invert the snap image (because the training dataset was inverted)
    c. reshape the photo to fit the model according to the training dataset.
