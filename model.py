import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from sklearn import preprocessing
import matplotlib.pyplot as plt

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

x_train = train.drop(['label'], axis=1).values
y_train = train['label'].values

x_test = test.drop(['label'], axis=1).values
y_test = test['label'].values

x_test.shape, y_test.shape

index = 2000

plt.imshow(x_train[index].reshape(50, 50), cmap='gray')
x_train = x_train.reshape(train.shape[0], 50, 50, 1).astype('float32')
x_train = x_train / 255.0

x_test = x_test.reshape(test.shape[0], 50, 50, 1).astype('float32')
x_test = x_test / 255.0

lb = preprocessing.LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("x_train shape", x_train.shape)
print("y_train shape", y_train.shape)

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu',
          input_shape=(50, 50, 1)))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(200))
model.add(Activation('relu'))

model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

model.summary()

model.compile(loss="categorical_crossentropy", optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=50, epochs=20, verbose=1)

predictions = model.evaluate(x_test, y_test)

# Sample Test
index = 100
plt.imshow(x_test[index].reshape(50, 50), cmap='gray')
print("Actual", y_test[index])
print("Predicted", model.predict([[x_test[index]]]))
