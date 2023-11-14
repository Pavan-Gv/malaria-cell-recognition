#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install seaborn


# In[2]:


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
# to share the GPU resources for multiple sessions
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True # dynamically grow the memory used on the GPU
config.log_device_placement = True # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


my_data_dir = 'dataset/cell_images'


# In[4]:


os.listdir(my_data_dir)


# In[5]:


test_path = my_data_dir+'/test/'
train_path = my_data_dir+'/train/'


# In[6]:


os.listdir(train_path)


# In[7]:


len(os.listdir(train_path+'/uninfected/'))


# In[8]:


len(os.listdir(train_path+'/parasitized/'))


# In[9]:


os.listdir(train_path+'/parasitized')[0]


# In[10]:


para_img= imread(train_path+
                 '/parasitized/'+
                 os.listdir(train_path+'/parasitized')[0])


# In[11]:


plt.imshow(para_img)


# In[12]:


dim1 = []
dim2 = []
for image_filename in os.listdir(test_path+'/uninfected'):
    img = imread(test_path+'/uninfected'+'/'+image_filename)
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)


# In[13]:


sns.jointplot(x=dim1,y=dim2)


# In[14]:


image_shape = (130,130,3)


# In[15]:


help(ImageDataGenerator)


# In[16]:


image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               width_shift_range=0.10, # Shift the pic width by a max of 5%
                               height_shift_range=0.10, # Shift the pic height by a max of 5%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )


# In[17]:


image_gen.flow_from_directory(train_path)


# In[18]:


image_gen.flow_from_directory(test_path)


# In[19]:


model = models.Sequential([
    layers.Input((130,130,3)),
    layers.Conv2D(32,kernel_size=3,activation="relu",padding="same"),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(32,kernel_size=3,activation="relu"),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(32,kernel_size=3,activation="relu"),
    layers.MaxPool2D((2,2)),
    layers.Flatten(),
    layers.Dense(32,activation="relu"),
    layers.Dense(1,activation="sigmoid")])
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[20]:


model.summary()


# In[21]:


batch_size = 16


# In[22]:


help(image_gen.flow_from_directory)


# In[23]:


train_image_gen = image_gen.flow_from_directory(train_path,
                                               target_size=image_shape[:2],
                                                color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary')


# In[24]:


train_image_gen.batch_size


# In[25]:


len(train_image_gen.classes)


# In[26]:


train_image_gen.total_batches_seen


# In[27]:


test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size=image_shape[:2],
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary',shuffle=False)


# In[28]:


train_image_gen.class_indices


# In[29]:


results = model.fit(train_image_gen,epochs=20,
                              validation_data=test_image_gen
                             )


# In[30]:


model.save('cell_model.h5')


# In[31]:


losses = pd.DataFrame(model.history.history)


# In[32]:


losses[['loss','val_loss']].plot()


# In[33]:


model.metrics_names


# In[34]:


model.evaluate(test_image_gen)


# In[35]:


pred_probabilities = model.predict(test_image_gen)


# In[36]:


test_image_gen.classes


# In[37]:


predictions = pred_probabilities > 0.5


# In[38]:


print(classification_report(test_image_gen.classes,predictions))


# In[39]:


confusion_matrix(test_image_gen.classes,predictions)


# In[ ]:




