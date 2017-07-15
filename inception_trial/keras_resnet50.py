
# coding: utf-8

# use Keras pre-trained VGG16
# ---------------------------
# this is my first notebook.
#
# pre-trained VGG16 is quickly and good performance.
#
# I learned from official Keras blog tutorial
# [Building powerful image classification models using very little data][1]
#
#
#   [1]: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

# ## resize train data and test data ##

# In[1]:

import matplotlib.pyplot as plt
# get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import cv2
import math
from glob import glob
import os

train_df = pd.read_csv("../input/train_v2.csv")

weather_labels = ['clear', 'partly_cloudy', 'haze', 'cloudy']
land_labels = ['primary', 'agriculture', 'water', 'cultivation', 'habitation', 'road']
rare_labels = ['slash_burn', 'conventional_mine', 'bare_ground',
               'artisinal_mine', 'blooming', 'selective_logging', 'blow_down']
other_labels = land_labels+rare_labels
labels = weather_labels+land_labels+rare_labels

#train = True
train = False
predict = True
#use_saved_model = ""
use_saved_model = "RESNET50_transferlearning_freeze_bot.model"
train_df = train_df[:15000]

if train:
  #tags to one hot vector
  tmp = train_df.tags.str.get_dummies(sep=" ")
  train_df = pd.concat( [train_df, tmp[labels] ], axis=1)

  #shuffle
  np.random.seed(0)
  train_df = train_df.reindex(np.random.permutation(tmp.index))

  img_path = "../input/train-jpg/"

  y_weather = train_df[weather_labels].values
  y_other   = train_df[land_labels+rare_labels].values

  x = np.zeros([len(train_df), 256,256,3], dtype='float32')
  for i,name in  enumerate(train_df.image_name):
      # read image
      img = img_path + name + ".jpg"
      img = cv2.imread(img)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      x[i] = img/256.

#
#
## In[4]:
#
#
#test_names = []
#file_paths = []
#
#for i in range(len(sample_submission)):
#    test_names.append(sample_submission.ix[i][0])
#    file_paths.append(img_path + str(int(sample_submission.ix[i][0])) + '.jpg')
#
#test_names = np.array(test_names)
#
#
## In[5]:
#
#test_images = []
#for file_path in file_paths:
#    # read image
#    img = cv2.imread(file_path)
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#    # resize
#    if(img.shape[0] > img.shape[1]):
#        tile_size = (int(img.shape[1]*256/img.shape[0]), 256)
#    else:
#        tile_size = (256, int(img.shape[0]*256/img.shape[1]))
#
#    # centering
#    img = centering_image(cv2.resize(img, dsize=tile_size))
#
#    # out put 224*224px
#    img = img[16:240, 16:240]
#    test_images.append(img)
#
#    path, ext = os.path.splitext(os.path.basename(file_paths[0]))
#
#test_images = np.array(test_images)
#
#
## save numpy array.
##
## Usually I separate code, data format and CNN.
#
## In[6]:
#
## np.savez('224.npz', x=x, y=y, test_images=test_images, test_names=test_names)
#
#
## ## split train data and validation data  ##
#
## In[7]:
#
## FIXME : do shuffleing on file name, before loading actual img
## tmp = tmp.reindex(np.random.permutation(tmp.index))
#data_num = len(y)
#random_index = np.random.permutation(data_num)
#
#x_shuffle = []
#y_shuffle = []
#for i in range(data_num):
#    x_shuffle.append(x[random_index[i]])
#    y_shuffle.append(y[random_index[i]])
#
#x = np.array(x_shuffle)
#y = np.array(y_shuffle)
#
#
## In[8]:
#
  val_split_num = int(round(0.2*len(x)))
  x_train = x[val_split_num:]
  x_val  = x[:val_split_num]
  y_weather_train = y_weather[val_split_num:]
  y_weather_val   = y_weather[:val_split_num]
  y_other_train = y_other[val_split_num:]
  y_other_val   = y_other[:val_split_num]
#
#print('x_train', x_train.shape)
#print('y_train', y_train.shape)
#print('x_test', x_test.shape)
#print('y_test', y_test.shape)
#
## In[9]:
#
#
#
## use Keras pre-trained RESNET50
## ---------------------------
##
## but kaggle karnel is not run
#
## In[10]:
#
from keras.models import Sequential, Model, load_model
from keras import applications
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense
#
img_rows, img_cols, img_channel = 256, 256, 3
#
#load_saved_model = True
#
if use_saved_model:
    model = load_model(use_saved_model)
else:
    base_model = applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(
            img_rows,
            img_cols,
            img_channel))

    #Freeze weight for lower layers
    for aLayer in base_model.layers:
        aLayer.trainable = False


#    # In[11]:
#
    #add_other_model = Sequential(name='other_output')
    #add_other_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    #add_other_model.add(Dense(256, activation='relu'))
    #add_other_model.add(Dense(len(land_labels+rare_labels), activation='sigmoid'))

    #add_weather_model = Sequential(name='weather_output')
    #add_weather_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    #add_weather_model.add(Dense(256, activation='relu'))
    #add_weather_model.add(Dense(len(weather_labels), activation='sigmoid'))

    tmp = Flatten()(base_model.output)
    tmp = Dense(256, activation='relu')(tmp)
    weather_output = Dense(len(weather_labels),
                           activation='sigmoid',
                           name='weather_output')(tmp)

    tmp = Flatten()(base_model.output)
    tmp = Dense(256, activation='relu')(tmp)
    other_output = Dense(len(land_labels+rare_labels),
                         activation='sigmoid',
                         name='other_output')(tmp)

    model = Model(inputs=base_model.input,
                  outputs= [
                            #add_weather_model(base_model.output),
                            #add_other_model(base_model.output),
                            weather_output,
                            other_output,
                           ]
                 )

    model.compile(
        loss={'weather_output': 'categorical_crossentropy', 'other_output': 'binary_crossentropy'},
        #loss='binary_crossentropy',
        #optimizer=optimizers.Adam(),
        optimizer=optimizers.SGD(
            lr=1e-4,
            momentum=0.9),
            metrics=['accuracy']
        )
#
model.summary()
#
#
## In[12]:
#

if train:
    from keras.preprocessing.image import ImageDataGenerator
    from keras.callbacks import ModelCheckpoint
    from keras.callbacks import EarlyStopping

    batch_size = 24
    epochs = 20

    train_datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True)

    #fit only required if enabled featurewise_center or featurewise_std_normalization or zca_whitening
    #train_datagen.fit(x_train)

    checkpoint = ModelCheckpoint('RESNET50_transferlearning_freeze_bot.model',
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True
                                )

    earlystop  = EarlyStopping(monitor='val_loss',
                               min_delta=0.0002,
                               patience=5, verbose=0, mode='auto')

    callbacks_list = [checkpoint, earlystop]

    def train_datagen_labeled():
        img_gen = train_datagen.flow( x_train,
                                      np.arange(x_train.shape[0]), #row index as label
                                      batch_size=batch_size,
                                      shuffle=False
                                    )
        while True:
            imgs,idxes = img_gen.next()
            weather = y_weather_train[idxes]
            other = y_other_train[idxes]
            yield (imgs, {'weather_output':weather,'other_output':other})





    history = model.fit_generator(
        train_datagen_labeled(),
        steps_per_epoch=x_train.shape[0] // batch_size,
        epochs=epochs,
        validation_data=(x_val,
                         {'weather_output':y_weather_val,'other_output':y_other_val}
                        ),
        callbacks=[checkpoint, earlystop]
    )


if predict:
    test_df = pd.read_csv("../input/sample_submission_v2.csv")
    #test_df = test_df[0:200]
    img_path = "../input/test-jpg-additional/"

    tags_col = []
    batch_size = 32
    threshold = 0.5
    x =  np.zeros( [batch_size, 256, 256, 3], dtype='float32')
    batch_id = 0
    while (batch_id*batch_size) < len(test_df):
        for i in range(batch_size):
            pos = batch_id*batch_size+i
            name = test_df.image_name[pos]
            # read image
            img = img_path + name + ".jpg"
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x[i] = img / 255.
            if (pos+1)>=len(test_df): break
        weather_vec, other_vec = model.predict(x)
        for j in range( i+1 ):
            tags = []
            tags.append( weather_labels[weather_vec[j].argmax()] )
            for idx,aProb in enumerate(other_vec[j]):
                if aProb > threshold: tags.append( other_labels[idx] )
            tags_col.append( " ".join(tags) )
        print ("Batch %s done" % batch_id)
        batch_id+=1
    test_df.tags = tags_col
    test_df.to_csv("submit_resnet50_freeze_bot.csv", index=False)
#
#
## ## predict test data ##
#
## In[15]:
#
#test_images = test_images.astype('float32')
#test_images /= 255
#
#
## In[16]:
#
#predictions = model.predict(test_images)
#
#
## In[ ]:
#
#history.history["val_acc"]
#
#
## In[ ]:
#
#sample_submission = pd.read_csv("../input/sample_submission.csv")
#
#for i, name in enumerate(test_names):
#    sample_submission.loc[
#        sample_submission['name'] == name,
#     'invasive'] = predictions[i]
#
#sample_submission.to_csv("submit_resnet50_batch6_AdamFineTune.csv", index=False)
#
#
## What to do next?
## ----------------
##
## I will try pre-trained ResNet, fine tune ResNet.
##
## This idea seems to be helpful.
##
## [Dogs vs. Cats Redux Playground Competition, 3rd Place Interview][1]
##
##
##   [1]: http://blog.kaggle.com/2017/04/20/dogs-vs-cats-redux-playground-competition-3rd-place-interview-marco-lugo/
