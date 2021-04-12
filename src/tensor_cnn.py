import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#fetch shape sizes
train_shape = x_train.shape[0]
test_shape = x_test.shape[0]
box_hw = 28 #dimensions of mnist are 28x28

#reshape to feed into keras
x_train = x_train.reshape(train_shape, box_hw, box_hw, 1)
x_test = x_test.reshape(test_shape, box_hw, box_hw, 1)
input_shape = (box_hw, box_hw, 1)

#convert our values to floats for normalization
x_train = x_train.astype(np.float_)
x_test = x_test.astype(np.float_)

#normalize our values between 0,1
x_train /= 255
x_test /= 255

#to make our data catgorizable, we use keras' to_categorical
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# To run on CPU
# with tf.device('/CPU:0'):

model = Sequential()
model.add(Conv2D(32, kernel_size=(8, 8), activation='relu', input_shape=input_shape))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.RMSprop(),metrics=['accuracy'])
model.fit(x_train, y_train,batch_size=32,epochs=10,validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test)

print('Test loss:', score[0])
print('Test accuracy:', score[1])