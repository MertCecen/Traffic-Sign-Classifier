import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

data = []
labels = []
number_of_classes = 43
current_path = os.getcwd() # get current working directory


out = False
for i in range(number_of_classes):
    path = os.path.join(current_path,"Train",str(i))
    images = os.listdir(path)
    
    for a in images:
        try:
            image = Image.open(path + '\\'+ a)
            
        except:
            print("Error while opening image")
            out = True 
            break
        
        try:
            image = image.resize((30,30))
        except:
            print("error while resizing image")
            out = True 
            break
        
        try:
            image = np.array(image)
            
        except:
            print("error np.array")
            out = True
            break
        
        try:
            data.append(image)
        except:
            print("error data.append")
            out = True
            break
        
        try:
            labels.append(i)
        except:
            print("error labels.append")
            out = True
            break
    if out:
        break
        

#Converting lists into numpy arrays
data = np.array(data)
labels = np.array(labels)

print(data.shape, labels.shape)
#Splitting training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, random_state = 42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Converting the labels into one hot encoding therefore use loss = categorical_crossentropy
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

# Build the model
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5), activation = "relu", input_shape = X_train.shape[1:]))
model.add(Conv2D(32, 5, activation = "relu"))
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(rate = 0.25))

model.add(Conv2D(64, 3, activation = "relu"))
model.add(Conv2D(64, 3, activation = "relu"))
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(rate = 0.25))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(rate = 0.5))
model.add(Dense(43, activation = "softmax"))

#Compilation of the model
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics=["accuracy"])
model.summary()
epochs = 15
batch_size = 64
history = model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, validation_data = (X_test, y_test)) # train model
print("Successfully trained the model")
# number of steps = 31367 (X_train) / batch_size (64) = 31367 / 64 = 491 steps
model.save("my_model.h5")
print("Saved model as my_model.h5")


#testing accuracy on test dataset
from sklearn.metrics import accuracy_score

y_test = pd.read_csv('Test.csv')

labels = y_test["ClassId"].values
imgs = y_test["Path"].values

data = []

for img in imgs:
    image = Image.open(img)
    image = image.resize((30,30))
    data.append(np.array(image))

X_test = np.array(data)

pred = model.predict_classes(X_test)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#Accuracy with the test data

print("Accuracy Score: ", accuracy_score(labels, pred))

model.save("traffic_classifier.h5")
