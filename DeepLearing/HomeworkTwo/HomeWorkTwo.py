import tensorflow as tf
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#data set import
import tensorflow.keras.datasets.mnist as mnist



#define a basic model using relu activation function and softmax for the output layer
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
    ])

#train model using adam and cross entropy since we are doing classification
model.compile(optimizer='adam',loss='crossentropy',metrics=['accuracy'])



#set up train,test, and validation data sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_test, x_val, y_train, y_val = train_test_split(
    x_test, y_test,
    test_size=0.2,
    random_state=42,
    stratify=y_test
)

model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=10,
    batch_size=32
)

print(model.evaluate(x_test, y_test))



