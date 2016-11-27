# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 13:57:06 2016

@author: michael
"""

# Abandon using clarifai api, instead use lasagne and theano to create my own cnn
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam
from PIL import Image
import glob
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import h5py


im = Image.open('album covers/Brutai_Born_ProgressiveMetal.jpg')
im = im.resize((96,96))
im = im.convert('L')
pixels = list(im.getdata())
pixels = pd.Series(pixels)
pixels = pixels/255.0
#width, height = im.size
#pixels = [pixels[i * width:(i+1) * width] for i in xrange(height)]

data1 = []
album_df = []

for file in glob.glob("album covers/*.jpg"):
    filename = file
    artist = filename[filename.find("/")+1:filename.find("_")]
    album = filename[filename.find("_")+1:filename.find("_",filename.find("_")+1)]
    style = filename[filename.find("_",filename.find("_")+1)+1:filename.find(".")]
    im = Image.open(filename).resize((96,96))
    im = im.convert('L')
    pixels = list(im.getdata())
    pixels = pd.Series(pixels)
    pixels = pixels/255.0
    data = [artist,album,style,filename,pixels]   
    data1.append(data)
album_df = pd.DataFrame(data1,columns=['Artist','Album','Style','Filename','Pixels'])

#train_data = album_df[['Style','Pixels']]

#img = train_data.loc[6].Pixels.reshape(96,96)
#plt.imshow(img,cmap='gray')

batch_size = 64
nb_classes = len(album_df.Style.unique())
nb_epoch = 200
img_rows, img_cols = 96,96
im_channels = 1
nb_filters = 32
nb_pool = 2
nb_conv = 3

Styles = list(enumerate(album_df.Style.unique()))
album_df.StyleCode= album_df.Style
Styles_dict = { name: i for i, name in Styles}
album_df.StyleCode = album_df.StyleCode.map( lambda x: Styles_dict[x]).astype(int)

lb=LabelBinarizer()
(X,y) = (album_df.Pixels,lb.fit_transform(album_df.StyleCode).astype(np.float32))

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=4)

X_train = np.vstack(X_train)
X_train = X_train.reshape(X_train.shape[0],1,96,96)

X_test = np.vstack(X_test)
X_test = X_test.reshape(X_test.shape[0],1,96,96)

net1 = Sequential()
net1.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='valid', input_shape=(1,img_rows,img_cols)))
convout1 = Activation('relu')
net1.add(convout1)
net1.add(Convolution2D(nb_filters, nb_conv, nb_conv))
convout2 = Activation('relu')
net1.add(convout2)
net1.add(MaxPooling2D(pool_size=(nb_pool,nb_pool)))
net1.add(Dropout(0.25))

net1.add(Flatten())
net1.add(Dense(128))
net1.add(Activation('relu'))
net1.add(Dropout(0.5))
net1.add(Dense(nb_classes))
net1.add(Activation('softmax'))
net1.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])


X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)        

net1.fit(X_train, y_train,batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_data=(X_test,y_test))

score = net1.evaluate(X_test,y_test,verbose=0)

net1.predict_classes(X_test)


im_test = Image.open('test.jpg')
im_test = im_test.resize((96,96))
im_test = im_test.convert('L')
pixels_test = list(im_test.getdata())
pixels_test = pd.Series(pixels_test)
pixels_test = pixels_test/255.0

test = np.vstack(pixels_test)  
test = test.reshape(1,1,96,96)

x = test.reshape(96,96)
plt.imshow(x,cmap='gray')
  
net1.predict_classes(test)
pred = Styles[net1.predict_classes(test)]
pred



#SAVE MODEL
#json_model = net1.to_json()
#with open('model_architecture_net1.json','w') as json_file:
#    json_file.write(json_model)
#net1.save_weights('model_weights.h5',overwrite=True)

#LOAD MODEL
#json_file = open('model_architecture_net1.json','r')
#loaded_model_json = json_file.read()
#json_file.close()
#net1 = model_from_json(loaded_model_json)
#net1.load_weights('model_weights.h5')
#
#net1.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
#score = net1.evaluate(X_test,y_test,verbose=0)
#
#a = net1.predict(test)
#a.argmax()


