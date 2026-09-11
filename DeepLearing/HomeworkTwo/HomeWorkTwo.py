from pathlib import Path

import tensorflow as tf
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#data set import
import tensorflow.keras.datasets.mnist as mnist



'''
Data Set
1. Load Data
2. Transform
3. Split into train, test, and validation sets
'''

#set up train,test, and validation data sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Flatten each 28x28 image into 784 inputs and scale pixels to [0, 1].
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

#Split the data
x_test, x_val, y_test, y_val = train_test_split(
    x_test, y_test,
    test_size=0.2,
    random_state=42,
    stratify=y_test
)

'''
Model One: Basic Architecture
1. Build model
2. Compile model
3. Train model
'''

#define a basic model using relu activation function and softmax for the output layer
model = tf.keras.Sequential([
    tf.keras.Input(shape=(784,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
    ])

#train model using adam and cross entropy since we are doing classification
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#train the model
model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=10,
    batch_size=32
)

'''
Model One: Confusion Matrix
1. Evaluate model
2. Get predicted classes
3. Construct confusion matrix
4. Save chart
'''

print(model.evaluate(x_test, y_test))

# Get probabilites from the test set
predicted_probabilities = model.predict(x_test, verbose=0)
#get the predicted class (the class with the highest probability)
predicted_classes = np.argmax(predicted_probabilities, axis=1)
#compare the predicted classes to the true classes and create a confusion matrix
class_counts = confusion_matrix(y_test, predicted_classes, labels=np.arange(10))

#Construct the confusion matrix heatmap
plt.figure(figsize=(10, 8))
sb.heatmap(
    class_counts,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=np.arange(10),
    yticklabels=np.arange(10),
    cbar_kws={'label': 'Number of images'},
)
plt.xlabel('Predicted class')
plt.ylabel('Correct class')
plt.title('Model One: Correct vs. Predicted Class')
plt.tight_layout()

#Save it to a file
chart_path = Path(__file__).with_name('confusion_matrix.png')
plt.savefig(chart_path, dpi=150)
print(f'Confusion matrix saved to {chart_path}')
plt.show()

'''
Model Two: Different Activation Functions
1. Build model
2. Compile model
3. Train model
'''

#define the model using a combination of activaton function
model = tf.keras.Sequential([
    tf.keras.Input(shape=(784,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(64, activation='sigmoid'),
    tf.keras.layers.Dense(32, activation='sigmoid'),
    tf.keras.layers.Dense(10, activation='softmax'),
    ])

#train model using adam and cross entropy since we are doing classification
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#train the model
model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=10,
    batch_size=32
)


'''
Model Two: Confusion Matrix
1. Evaluate model
2. Get predicted classes
3. Construct confusion matrix
4. Save chart
'''

print(model.evaluate(x_test, y_test))

# Get probabilites from the test set
predicted_probabilities = model.predict(x_test, verbose=0)
#get the predicted class (the class with the highest probability)
predicted_classes = np.argmax(predicted_probabilities, axis=1)
#compare the predicted classes to the true classes and create a confusion matrix
class_counts = confusion_matrix(y_test, predicted_classes, labels=np.arange(10))

#Construct the confusion matrix heatmap
plt.figure(figsize=(10, 8))
sb.heatmap(
    class_counts,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=np.arange(10),
    yticklabels=np.arange(10),
    cbar_kws={'label': 'Number of images'},
)
plt.xlabel('Predicted class')
plt.ylabel('Correct class')
plt.title('Model Two: Correct vs. Predicted Class')
plt.tight_layout()

#Save it to a file
chart_path = Path(__file__).with_name('confusion_matrix_model_two.png')
plt.savefig(chart_path, dpi=150)
print(f'Confusion matrix saved to {chart_path}')
plt.show()

'''
Model Three;
'''
