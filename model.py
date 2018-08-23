import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D,Lambda,Cropping2D
from keras.callbacks import LearningRateScheduler
def learning_rate(epoch):
    return 0.001*(0.1**int(epoch/10))

samples = []
correction=0.15

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

images=[]
angles=[]

for batch_sample in samples:       
    
    cenimage = cv2.imread(batch_sample[0])
    center_image=cv2.cvtColor(cenimage,cv2.COLOR_BGR2RGB)
    limage=cv2.imread(batch_sample[1])
    left_image=cv2.cvtColor(limage,cv2.COLOR_BGR2RGB)
    rimage=cv2.imread(batch_sample[2])
    right_image=cv2.cvtColor(rimage,cv2.COLOR_BGR2RGB)

    center_angle = float(batch_sample[3])
    left_angle=center_angle+correction
    right_angle=center_angle-correction
    #images.append(center_image)
    #angles.append(center_angle)
    images.extend([center_image,left_image,right_image])               
    angles.extend([center_angle,left_angle,right_angle])

def flip(image):
    return np.fliplr(image)

for i in range(len(images)):
    image1=flip(images[i])
    images.append(image1)
    angles.append(-1*angles[i])
    
X_train = np.array(images)
y_train = np.array(angles)

X_train,y_train=shuffle(X_train,y_train)


model=Sequential()
model.add(Cropping2D(cropping=((50,20), (0,10)),input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

model.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2)))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(100,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(50,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1))
model.load_weights('weights.h5')
model.compile(loss='mse', optimizer='adam')

logits=model.fit(X_train,y_train,epochs=20,shuffle=True,validation_split=0.2,callbacks=[LearningRateScheduler(learning_rate)])
model.save_weights('weights.h5')
model.save('model.h5')

plt.plot(logits.history['loss'])
plt.plot(logits.history['val_loss'])
plt.title('Training loss vs Validation loss')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.legend(['Train','Validation'],loc='upper right')
plt.savefig('Learning.jpg')