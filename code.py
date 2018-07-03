import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
from keras import optimizers

def one_hot_encode_object_array(arr):
    '''One hot encode a numpy array of objects (e.g. strings)'''
    uniques, ids = np.unique(arr, return_inverse=True)
    return np_utils.to_categorical(ids, len(uniques))

fields = ['dataResult','HomeWin','Draw','AwayWin']
traindata =pd.read_csv('17-18.csv', usecols=fields)
train_X = traindata.values[:, 1:4]
train_Y = traindata.values[:, 0]
train_y_ohe = one_hot_encode_object_array(train_Y)
print(train_y_ohe)
testdata =pd.read_csv('16-17.csv', usecols=fields)

test_X = testdata.values[:, 1:4]
test_Y = testdata.values[:, 0]
test_y_ohe = one_hot_encode_object_array(test_Y)

model = Sequential()
model.add(Dense(16, input_shape=(3,)))
model.add(Activation('relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=["accuracy"])
model.fit(train_X, train_y_ohe, epochs=100, batch_size=1, verbose=1, validation_data=(test_X, test_y_ohe))
loss, accuracy = model.evaluate(test_X, test_y_ohe, verbose=1)
print("Accuracy = {:.2f}".format(accuracy))
y_pred = model.predict(train_X)