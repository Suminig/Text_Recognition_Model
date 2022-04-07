#%%
#Import Important Libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
from sklearn.model_selection import train_test_split
import seaborn as sns

import os
#%%
#Import datasets

testing = pd.read_csv('./input/emnist-balanced-test.csv')
training = pd.read_csv('./input/emnist-balanced-train.csv')
#%%

#training_letters
y1 = np.array(training.iloc[:,0].values)
x1 = np.array(training.iloc[:,1:].values)
#testing_letters
y2 = np.array(testing.iloc[:,0].values)
x2 = np.array(testing.iloc[:,1:].values)

#%%
# Show examples from the dataset
rows = 5 
cols = 5 
mapping = ['0','1','2','3','4','5','6','7','8','9',
            'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
            'a','b','d','e','f','g','h','n','q','r','t']
f = plt.figure(figsize=(2*cols,2*rows))  

for i in range(rows*cols):
    cur_digit = x1[i].reshape([28,28]).transpose()
    f.add_subplot(rows,cols,i+1) 
    plt.imshow(cur_digit,cmap="Greys") 
    plt.axis("off")
    plt.title(mapping[y1[i]])
plt.savefig("transposed_digits.png")

#%% 
    
# Normalise and reshape data
train_images = x1 / 255.0
test_images = x2 / 255.0

for i in range(train_images.size):
    train_images = train_images.transpose()

train_images_number = train_images.shape[0]
train_images_height = 28
train_images_width = 28
train_images_size = train_images_height*train_images_width

train_images = train_images.reshape(train_images_number, train_images_height, train_images_width, 1)
    
test_images_number = test_images.shape[0]
test_images_height = 28
test_images_width = 28
test_images_size = test_images_height*test_images_width

test_images = test_images.reshape(test_images_number, test_images_height, test_images_width, 1)
#%%
# 47 Distinct outputs in EMNIST Balanced Dataset
number_of_classes = 47

y1 = tf.keras.utils.to_categorical(y1, number_of_classes)
y2 = tf.keras.utils.to_categorical(y2, number_of_classes)

train_x,test_x,train_y,test_y = train_test_split(train_images,y1,test_size=0.2,random_state = 42)
#%%
# Model Designing
model = tf.keras.Sequential([ 
    tf.keras.layers.Conv2D(32,3,input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(input_shape=(28,28,1)),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(number_of_classes,activation='softmax')
])

print(model.summary()) 
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])


# Model Training
MCP = ModelCheckpoint('Best_points.h5',verbose=1,save_best_only=True,monitor='val_accuracy',mode='max')
ES = EarlyStopping(monitor='val_accuracy',min_delta=0,verbose=0,restore_best_weights = True,patience=3,mode='max')

history = model.fit(train_x,train_y,epochs=10,validation_data=(test_x,test_y),callbacks=[MCP,ES])

#%%

# Graph - Accuracy vs. Validation_Accuracy
q = len(history.history['accuracy'])

plt.figsize=(10,10)
sns.lineplot(x = range(1,1+q),y = history.history['accuracy'], label='Accuracy')
sns.lineplot(x = range(1,1+q),y = history.history['val_accuracy'], label='Val_Accuracy')
plt.xlabel('epochs')
plt.ylabel('Accuray')

#%%

# Model Evaluating
score = model.evaluate(test_images, y2, verbose=1, batch_size=1)
print("Loss: %.2f%%"%(score[0]*100)+" / Accuracy: %.2f%%"%(score[1]*100))



#%%

# Saves the model into as json file, h5 file

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
    
model.save_weights("model.h5")

#%%

# Saves the model into .tflite file
converter = tf.lite.TFLiteConverter.from_keras_model(model)
model = converter.convert()

file = open('model.tflite' , 'wb' ) 
file.write(model)



