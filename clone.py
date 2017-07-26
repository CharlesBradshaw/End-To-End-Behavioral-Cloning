import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

lines = []

with open('../data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []

data = []
for line in lines:
	data.append(line)

import sklearn
from sklearn.model_selection import train_test_split
train_data, validation_data = train_test_split(data, test_size=0.2)

def generator(data, batch_size=36):
    data_len = len(data)
    correction = [0,0.3,-0.3]
    path = "../data/IMG/"
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(data)
        for offset in range(0, data_len, int(batch_size/6)):
            batch_lines = data[offset:offset+int(batch_size/6)]
            images = []
            angles = []

            for batch_line in batch_lines:
            	angle = batch_line[3]
            	for i in range(3):
	                new_path = path + batch_line[i].split('/')[-1]
	                img = cv2.imread(new_path)
	                corrected_angle = float(angle) + correction[i]
	                images.append(img)
	                angles.append(corrected_angle)

	                images.append(cv2.flip(img,1))
	                angles.append(-corrected_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)



train_generator = generator(train_data, batch_size=36)
validation_generator = generator(validation_data, batch_size=36)

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

input_shape = (160,320,3)
model = Sequential()

model.add(Lambda(lambda x: x / 255 - 0.5, input_shape = input_shape))
model.add(Cropping2D(cropping = ((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample = (2,2),activation = "relu"))
model.add(Convolution2D(36,5,5,subsample = (2,2),activation = "relu"))
model.add(Convolution2D(48,5,5,subsample = (2,2),activation = "relu"))
model.add(Convolution2D(64,3,3,activation = "relu"))
model.add(Convolution2D(64,3,3,activation = "relu"))

model.add(Flatten())
model.add(Dense(100 , activation='relu'))
model.add(Dense(50,  activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_data), validation_data = \
		validation_generator, nb_val_samples=len(validation_data),nb_epoch=3)

model.save('model.h5')