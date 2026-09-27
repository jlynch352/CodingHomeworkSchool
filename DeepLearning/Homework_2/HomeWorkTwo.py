#import libs
from pathlib import Path
import tensorflow as tf
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#data set import
import tensorflow.keras.datasets.mnist as mnist

#Builds directory path
image_dir = Path(__file__).resolve().parent / 'images'
#Create the directory regardless of whether it already exists
image_dir.mkdir(exist_ok=True)

'''
Data Set
1. Load Data
2. Transform
3. Split into train, test, and validation sets
'''

#set up train,test, and validation data sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Flatten each 28x28 image into 784 inputs and rescale
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# Split the training data; keep the original test set for final evaluation.
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
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

# Stop after 3 epochs without improved validation loss and restore the best weights.
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Save the history so we can report how many epochs actually ran.
history_one = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=10,
    batch_size=32,
    callbacks=[early_stopping]
)

'''
Model One: Confusion Matrix
1. Evaluate model
2. Get predicted classes
3. Construct confusion matrix
4. Save chart
'''

model_one_loss, model_one_accuracy = model.evaluate(x_test, y_test)

# Get probabilites from the test set
predicted_probabilities = model.predict(x_test, verbose=0)
#get the predicted class (the class with the highest probability)
predicted_classes = np.argmax(predicted_probabilities, axis=1)
#compare the predicted classes to the true classes and create a confusion matrix
class_counts = confusion_matrix(y_test, predicted_classes, labels=np.arange(10))
model_one_counts = class_counts.copy()

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
chart_path = image_dir / 'confusion_matrix_one.png'
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
    tf.keras.layers.Dense(32, activation='leaky_relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
    ])

#train model using adam and cross entropy since we are doing classification
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# Stop after 3 epochs without improved validation loss and restore the best weights.
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Save the history so we can report how many epochs actually ran.
history_two = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=10,
    batch_size=32,
    callbacks=[early_stopping]
)


'''
Model Two: Confusion Matrix
1. Evaluate model
2. Get predicted classes
3. Construct confusion matrix
4. Save chart
'''

model_two_loss, model_two_accuracy = model.evaluate(x_test, y_test)

# Get probabilites from the test set
predicted_probabilities = model.predict(x_test, verbose=0)
#get the predicted class (the class with the highest probability)
predicted_classes = np.argmax(predicted_probabilities, axis=1)
#compare the predicted classes to the true classes and create a confusion matrix
class_counts = confusion_matrix(y_test, predicted_classes, labels=np.arange(10))
model_two_counts = class_counts.copy()

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
chart_path = image_dir / 'confusion_matrix_model_two.png'
plt.savefig(chart_path, dpi=150)
print(f'Confusion matrix saved to {chart_path}')
plt.show()

'''
Model Three: Fewer, wider hidden layers with a higher training limit
1. Build model
2. Compile model
3. Train model
'''

#define the model using a combination of activaton function
model = tf.keras.Sequential([
    tf.keras.Input(shape=(784,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(10, activation='softmax'),
    ])

#train model using adam and cross entropy since we are doing classification
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# Stop after 3 epochs without improved validation loss and restore the best weights.
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Save the history so we can report how many epochs actually ran.
history_three = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping]
)


'''
Model Three: Confusion Matrix
1. Evaluate model
2. Get predicted classes
3. Construct confusion matrix
4. Save chart
'''

model_three_loss, model_three_accuracy = model.evaluate(x_test, y_test)

# Get probabilites from the test set
predicted_probabilities = model.predict(x_test, verbose=0)
#get the predicted class (the class with the highest probability)
predicted_classes = np.argmax(predicted_probabilities, axis=1)
#compare the predicted classes to the true classes and create a confusion matrix
class_counts = confusion_matrix(y_test, predicted_classes, labels=np.arange(10))
model_three_counts = class_counts.copy()

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
plt.title('Model Three: Correct vs. Predicted Class')
plt.tight_layout()

#Save it to a file
chart_path = image_dir / 'confusion_matrix_model_three.png'
plt.savefig(chart_path, dpi=150)
print(f'Confusion matrix saved to {chart_path}')
plt.show()


'''
Comparing The Three Models
1. Print test accuracy and loss (higher accuracy and lower loss are better)
2. Chart test accuracy
3. Compare digit errors and summarize the results
'''

#set up data for evaluation 
model_names = ['Model One', 'Model Two', 'Model Three']
accuracies = [model_one_accuracy, model_two_accuracy, model_three_accuracy]
losses = [model_one_loss, model_two_loss, model_three_loss]
histories = [history_one, history_two, history_three]
epoch_counts = [len(history.epoch) for history in histories]

# Print out accuracy and loss for each model 
print('\nComparing The Three Models')
print(f'{"Model":<15}{"Epochs":<10}{"Test Accuracy":<18}{"Test Loss":<12}')
for name, epochs, accuracy, loss in zip(model_names, epoch_counts, accuracies, losses):
    print(f'{name:<15}{epochs:<10}{accuracy * 100:>12.2f}%     {loss:<12.4f}')

#Construct bar chart to compare the test accuracy of the three models
plt.figure(figsize=(8, 5))
bars = plt.bar(model_names, [accuracy * 100 for accuracy in accuracies],color=['steelblue', 'orange', 'seagreen'])
plt.bar_label(bars, fmt='%.2f%%', padding=3)
plt.ylabel('Test accuracy (%)')
plt.ylim(0, 105)
plt.title('Comparing the Three Models: Test Accuracy')
plt.ylim(97, 98)
plt.tight_layout()
#Save it to a file
chart_path = image_dir / 'model_accuracy_comparison.png'
plt.savefig(chart_path, dpi=150)
print(f'Comparison chart saved to {chart_path}')
plt.show()

