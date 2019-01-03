
# coding: utf-8

# In[1]:

from keras.models import Sequential
from scipy.misc import imread
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import keras
import pandas as pd
import h5py
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import tarfile
from skimage.io import imread_collection
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions


# In[2]:

batch_size = 30
num_classes = 6
epochs = 5
#data_augmentation = True
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_intel_basic_trained_model1.h5'


# In[3]:

train=pd.read_csv('train-scene/train.csv')
test=pd.read_csv('test.csv')


# In[4]:

image_path='train-scene/train/'


# In[5]:

from scipy.misc import imresize
train_img=[]
for i in range(len(train)):
    temp_img=image.load_img(image_path+train['image_name'][i],target_size=(150,150))
    temp_img=image.img_to_array(temp_img)
    train_img.append(temp_img)


# In[6]:

x_train=np.array(train_img)
#train_img=preprocess_input(train_img)


# In[7]:

test_img=[]
for i in range(len(test)):
    temp_img=image.load_img(image_path+test['image_name'][i],target_size=(150,150))
    temp_img=image.img_to_array(temp_img)
    test_img.append(temp_img)


# In[8]:

test_img=np.array(test_img)
#test_img=preprocess_input(test_img)


# In[9]:

y_train = keras.utils.to_categorical(train[['label']], num_classes)


# In[10]:

model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)


# In[11]:

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
x_train = x_train.astype('float32')


# In[12]:

x_train /= 255
model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True)


# In[13]:

x_test = test_img.astype('float32')
x_test /= 255
classes = model.predict(x_test, batch_size=30)


# In[14]:

class_labels = np.argmax(classes, axis=1)
class_labels_dt=pd.DataFrame(class_labels)
class_labels_dt.columns=['label']


# In[15]:

frames=[test,class_labels_dt]
result = pd.concat(frames,axis=1)


# In[16]:

result.to_csv('intel_model_basic_5_64_128_150p.csv')


# In[18]:

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_intel_trained_model_5_64_128_150p.h5'


# In[19]:

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

