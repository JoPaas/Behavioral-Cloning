import os
import csv
import copy

samples = []
steering_offset = 0.15 # steering angle offset for off-center images
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        # read center, left and right images
        tmp_center = copy.copy(line)
        del tmp_center[1:3]

        # add offset to steering angles for left and right images
        tmp_1eft = copy.copy(line)
        del tmp_1eft[2]
        del tmp_1eft[0]
        tmp_1eft[1] = float(tmp_1eft[1])+steering_offset

        tmp_right = copy.copy(line)
        del tmp_right[0:2]
        tmp_right[1] = float(tmp_right[1]) - steering_offset

        samples.append(tmp_center)
        samples.append(tmp_1eft)
        samples.append(tmp_right)
print(len(samples))
print(samples[0:3])

import matplotlib.pyplot as plt
import numpy as np
angles = [float(sample[1]) for sample in samples]
n, bins, patches = plt.hist(angles, 23)
plt.show() # visualize histogram of steering angles


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.1)





import cv2
import numpy as np
import sklearn
from random import shuffle

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            # read image and convert to RGB since cv2 imports as BGR as default
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split("\\")[-1]
                image = cv2.imread(name)
                angle = float(batch_sample[1])
                imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(imageRGB)
                angles.append(angle)

            # output images
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 160, 320  # Trimmed image format

from keras.models import Sequential
from keras.layers import Lambda,Cropping2D,Convolution2D,Flatten,Dense,Conv2D

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation and crop input image
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((70,25), (0,0))))

model.add(Conv2D(24,5, strides=(2,2), activation='relu'))
model.add(Conv2D(36,5, strides=(2,2), activation='relu'))
model.add(Conv2D(48,5, strides=(2,2), activation='relu'))
model.add(Conv2D(64,3, activation='relu'))
model.add(Conv2D(64,3, activation='relu'))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                    validation_data=validation_generator, nb_val_samples=len(validation_samples),
                    nb_epoch=3)

model.save('model.h5')

import matplotlib.pyplot as plt

print('###### Training Statistics ######')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
