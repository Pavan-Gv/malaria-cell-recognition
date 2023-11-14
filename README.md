# Deep Neural Network for Malaria Infected Cell Recognition

## AIM

To develop a deep neural network for Malaria infected cell recognition and to analyze the performance.

## Problem Statement and Dataset
The Convolutional Neural Network method reduces the likelihood of overfitting by employing data augmentation. Thus, it was found that deep learning-based malaria detection systems outperformed most conventional methods in speed. To distinguish between the images of parasitized and uninfected smear blood cells, a convolutional neural network was created and taught via training. CNN is capable of extracting three types of image features: low-level, mid-level, and high-level features. These functions are used to extract the traditional picture features.
## Neural Network Model
<img width="748" alt="image" src="https://github.com/KoduruSanathKumarReddy/malaria-cell-recognition/assets/69503902/ad37b250-512d-45a7-babf-6fce8c133c65">



## DESIGN STEPS:

**STEP 1:** Import tensorflow and preprocessing libraries
**STEP 2:** Create an ImageDataGenerator to flow image data
**STEP 3:** Build the convolutional neural network model and train the model
**STEP 4:** Evaluate the model with the testing data
**STEP 5:** Fit the model
**STEP 6:** Plot the performance plot

## PROGRAM

~~~
Developed by: G Venkata Pavan Kumar
Registered No: 212221240013
~~~
~~~
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
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

my_data_dir = 'dataset/cell_images'
os.listdir(my_data_dir)
test_path = my_data_dir+'/test/'
train_path = my_data_dir+'/train/'
os.listdir(train_path)
len(os.listdir(train_path + '/uninfected/'))
len(os.listdir(train_path + '/parasitized/'))
os.listdir(train_path + '/parasitized')[0]


para_img = imread(train_path + '/parasitized/' + os.listdir(train_path + '/parasitized')[0])
para_img.shape
plt.imshow(para_img)

dim1 = []
dim2 = []
for image_filename in os.listdir(test_path + '/uninfected'):
    img = imread(test_path + '/uninfected' + '/' + image_filename)
    d1, d2, colors = img.shape
    dim1.append(d1)
    dim2.append(d2)
sns.jointplot(x=dim1, y=dim2)

image_shape = (130, 130, 3)
image_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.10,
    height_shift_range=0.10,
    rescale=1/255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

image_gen.flow_from_directory(train_path)
image_gen.flow_from_directory(test_path)

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

train_image_gen = image_gen.flow_from_directory(train_path, target_size=image_shape[:2], color_mode='rgb',
                                                batch_size=16, class_mode='binary')

train_image_gen.batch_size
len(train_image_gen.classes)
train_image_gen.total_batches_seen

test_image_gen = image_gen.flow_from_directory(test_path, target_size=image_shape[:2], color_mode='rgb',
                                              batch_size=16, class_mode='binary', shuffle=False)

train_image_gen.class_indices

results = model.fit(train_image_gen, epochs=5, validation_data=test_image_gen)
model.save('cell_model1.h5')
losses = pd.DataFrame(model.history.history)
losses.plot()
model.evaluate(test_image_gen)
pred_probabilities = model.predict(test_image_gen)
predictions = pred_probabilities > 0.5
print(classification_report(test_image_gen.classes, predictions))
confusion_matrix(test_image_gen.classes, predictions)
from tensorflow.keras.preprocessing import image

img = image.load_img('new.png')
img = tf.convert_to_tensor(np.asarray(img))
img = tf.image.resize(img, (130, 130))
img = img.numpy()

type(img)
plt.imshow(img)

x_single_prediction = bool(model.predict(img.reshape(1, 130, 130, 3)) > 0.6)
print(x_single_prediction)

if x_single_prediction == 1:
    print("Uninfected")
else:
    print("Parasitized")

~~~

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

<img width="685" alt="image" src="https://github.com/KoduruSanathKumarReddy/malaria-cell-recognition/assets/69503902/09210495-7666-4e00-970b-d5cd9c6038f8">


### Classification Report

<img width="685" alt="image" src="https://github.com/KoduruSanathKumarReddy/malaria-cell-recognition/assets/69503902/069114fd-3da0-4051-b20c-f5c784c56a1d">


### Confusion Matrix

<img width="241" alt="image" src="https://github.com/KoduruSanathKumarReddy/malaria-cell-recognition/assets/69503902/878aed26-77bf-4355-b393-0f7df709885a">


### New Sample Data Prediction

<img width="351" alt="image" src="https://github.com/KoduruSanathKumarReddy/malaria-cell-recognition/assets/69503902/e38c627a-c3b9-483d-8a48-e68a7bcc8df4">


## RESULT
Thus, a deep neural network for Malaria infected cell recognized and analyzed the performance .
