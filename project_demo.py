# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 10:56:48 2018

@author: Nasif
"""

import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.externals import joblib
from scipy import misc
from PIL import Image
import PIL.ImageOps

import tkinter as tk
import subprocess
import os
from pdf2image import convert_from_path


class Paint(object):

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_BRUSH_SIZE = 8.0
    DEFAULT_COLOR = 'black'

    def __init__(self):
        self.root = tk.Tk()
        
#        self.pen_button = tk.Button(self.root, text='pen', command=self.use_pen)
#        self.pen_button.grid(row=0, column=0)

        self.brush_button = tk.Button(self.root, text='brush', command=self.use_brush)
        self.brush_button.grid(row=0, column=0)

#        self.color_button = tk.Button(self.root, text='color', command=self.choose_color)
#        self.color_button.grid(row=0, column=1)

        self.eraser_button = tk.Button(self.root, text='eraser', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=2)
        
        self.save_button = tk.Button(self.root, text='save', command=self.use_save)
        self.save_button.grid(row=0, column=4)
            
#        self.choose_size_button = tk.Scale(self.root, from_=1, to=10, orient=tk.HORIZONTAL)
#        self.choose_size_button.grid(row=0, column=4)

        self.c = tk.Canvas(self.root, bg='white', width=500, height=500)
        self.c.grid(row=1, columnspan=5)
       

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = 4 #self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.brush_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

#    def use_pen(self):
#        self.activate_button(self.pen_button)

    def use_brush(self):
        self.activate_button(self.brush_button)

#    def choose_color(self):
#        self.eraser_on = False
#        self.color = tk.colorchooser.askcolor(self.color)[1]

    def use_eraser(self):
        self.activate_button(self.c , eraser_mode=True)
        
    def use_save(self):
        self.c.update()   
        self.c.postscript(file="draw.ps", colormode='color')
        
        process = subprocess.Popen(["ps2pdf", "draw.ps", "result.pdf"], shell=True)
        process.wait()
        os.remove("draw.ps")
        
        images = convert_from_path("result.pdf")
        os.remove("result.pdf")
        
#        images[0].crop((400, 500, 1400, 1500)).save("cur.jpg")
        images[0].save("cur.jpg")
        
        self.root.destroy()
#        self.root.quit()
        

    #TODO: reset canvas
    #TODO: undo and redo
    #TODO: draw triangle, rectangle, oval, text

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=tk.RAISED)
        some_button.config(relief=tk.SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.line_width = 8 #self.choose_size_button.get()
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y, width=self.line_width,
                               fill=paint_color, capstyle=tk.ROUND, smooth=tk.TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None


if __name__ == '__main__':
    ge = Paint()
#
#
#digits = datasets.load_digits()
#features = digits.data 
#labels = digits.target
#
#clf = SVC(gamma = 0.001)
#clf.fit(features, labels)
    
clf=joblib.load('traindata.pkl')

image = Image.open('cur.jpg')
if image.mode == 'RGBA':
    r,g,b,a = image.split()
    rgb_image = Image.merge('RGB', (r,g,b))

    inverted_image = PIL.ImageOps.invert(rgb_image)

    r2,g2,b2 = inverted_image.split()

    final_transparent_image = Image.merge('RGBA', (r2,g2,b2,a))
    
    final_transparent_image.save('cur1.jpg')

else:
    inverted_image = PIL.ImageOps.invert(image)
    inverted_image.save('cur1.jpg')
    
img = misc.imread("cur1.jpg") 
img = misc.imresize(img, (28, 28))

#for index, (image, label) in enumerate(zip(train_img[0:5], train_lbl[0:5])):
#    plt.subplot(1, 5, index + 1)

plt.imshow(img, cmap=plt.cm.gray)

#    plt.title('Training: %i\n' % label, fontsize = 20)
#img.save("crop.jpg")
#img = img.astype(digits.images.dtype)
#img = misc.bytescale(img, high=16, low=0)



x_test = []

for eachRow in img:
	for eachPixel in eachRow:
		x_test.append(sum(eachPixel/3.0))


print(clf.predict([x_test]))


#x=input("enter any key")
