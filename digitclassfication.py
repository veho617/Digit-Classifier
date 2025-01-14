#digit recognition
#data set beingf used is built i to keras, has 60000 28x28 grayscale images.
#x are the images and y is the outputs/labels

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras import layers
from keras import Sequential
from keras import losses
import random

#loading the dataset
(x_train , y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path="mnist.npz")

#data preprocessing, kind of like feature scaling
x_train = x_train/255.0
x_test = x_test/255.0


#defining the model architecture

model = Sequential([
    layers.Flatten(input_shape = (28 , 28)),
    layers.Dense(25, activation = "relu"),
    layers.Dense(15, activation = "relu"),
    layers.Dense(10, activation = "softmax"),
])


#training the model
model.compile(optimizer="adam",
              loss = losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
model.fit(x_train,y_train,epochs=10) 

#test the model
test_loss , test_acc = model.evaluate(x_test,y_test,verbose = 1)
print("Test accuracy is ", test_acc)
print("Test loss", test_loss)

#use the model to predict

n = int(input("Enter number of images you want to predict from the dataset: "))
test_index = 0
for i in range(n):
    test_index = random.randrange(0,10000)
    prediction = model.predict(x_test[test_index].reshape(1,28,28))
    print("Prediction is ", np.argmax(prediction))
    print("The actual label is ", y_test[test_index])
    plt.imshow(x_test[test_index], cmap='gray')
    plt.show()