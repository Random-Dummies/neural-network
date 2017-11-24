
from __future__ import print_function

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import matplotlib.pyplot as plt
from keras.models import model_from_json

import pandas as pd
df=pd.read_csv("ldata.csv",header = None,sep = ",")
data = df.as_matrix()
x_train = data[:,:5]
y_train = np.zeros((100,1))
y_train[:,0] = data[:,5]
print(y_train.shape)
y_train= keras.utils.to_categorical(y_train, num_classes=10)

# print(y_train)
print('Building model...')
model = Sequential()
model.add(Dense(5, input_shape=(5,)))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=5,
                    epochs=2000,
                    verbose=1,
                    validation_split=0.1)
score = model.evaluate(x_train, y_train,
                       batch_size=5, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])
plt.plot(history.history['loss'])
plt.show()
model_json = model.to_json()
with open("model_nucthon.json", "w") as json_file:
    json_file.write(model_json)
	# serialize weights to HDF5
model.save_weights("model_nucthon.h5")
print("Saved model to disk")
