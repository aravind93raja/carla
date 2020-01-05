from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten , Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
import random
from matplotlib import pyplot


model = Sequential()

model.add(Conv2D(24, 5, 5, subsample=(2, 2), border_mode="same", input_shape=(50,50,1)))
model.add(Activation('relu'))
model.add(Conv2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Activation('relu'))
model.add(Conv2D(48, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Activation('relu'))
model.add(Conv2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
model.add(Activation('relu'))
model.add(Conv2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
model.add(Flatten())

model.add(Dropout(0.5))

model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))

print (model.summary())

model.compile(loss='mse',optimizer=Adam(1e-5),metrics=['mae'])

data_x=np.load("/home/deeplearner/UnrealEngine_4.18/carla/PythonClient/Dataset/data_x .npy")
data_y=np.load("/home/deeplearner/UnrealEngine_4.18/carla/PythonClient/Dataset/data_y .npy")


filepath="weights-improvement-{epoch:02d}-{val_mae:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_mae', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]



#Fit the model
history=model.fit(data_x,data_y,validation_split=0.2,shuffle=True,epochs=50,batch_size=64,callbacks=callbacks_list,verbose=1)
model.save('/home/deeplearner/UnrealEngine_4.18/carla/PythonClient/regression.h5') 



#plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# plot mse during training
pyplot.subplot(212)
pyplot.title('Mae')
pyplot.plot(history.history['mae'], label='train')
pyplot.plot(history.history['val_mae'], label='test')
pyplot.legend()
pyplot.show()