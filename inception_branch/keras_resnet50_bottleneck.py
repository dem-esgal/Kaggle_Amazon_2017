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

train = True
#train = False
#predict = True
validate = True
predict = False

#use_saved_model = ""
use_saved_model = "RESNET50_224_bottleneck_top.model"

if train:
  #tags to one hot vector
  tmp = train_df.tags.str.get_dummies(sep=" ")
  train_df = pd.concat( [train_df, tmp[labels] ], axis=1)

  #shuffle
  np.random.seed(0)
  order = np.random.permutation(tmp.index)
  train_df = train_df.reindex(order)

  img_path = "../input/train-jpg/"

  y_weather = train_df[weather_labels].values
  y_other   = train_df[land_labels+rare_labels].values

  x = np.load(open('resnet50_features_train_jpg_224.npy', 'rb'))
  x = x[order]

  val_split_num = int(round(0.2*len(x)))
  x_train = x[val_split_num:]
  x_val  = x[:val_split_num]
  y_weather_train = y_weather[val_split_num:]
  y_weather_val   = y_weather[:val_split_num]
  y_other_train = y_other[val_split_num:]
  y_other_val   = y_other[:val_split_num]

from keras.models import Sequential, Model, load_model
from keras import applications
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense, Input

if use_saved_model:
    model = load_model(use_saved_model)
else:
    main_input = Input(shape=(x.shape[1],),name='main_input')

    tmp = Dense(256, activation='relu')(main_input)
    tmp = Dense(256, activation='relu')(tmp)
    weather_output = Dense(len(weather_labels),
                           activation='sigmoid',
                           name='weather_output')(tmp)

    tmp = Dense(256, activation='relu')(main_input)
    tmp = Dense(256, activation='relu')(tmp)
    other_output = Dense(len(land_labels+rare_labels),
                         activation='sigmoid',
                         name='other_output')(tmp)

    model = Model(inputs=main_input,
                  outputs= [
                            weather_output,
                            other_output,
                           ]
                 )

    model.compile(
        #loss={'weather_output': 'categorical_crossentropy', 'other_output': 'binary_crossentropy'},
        loss={'weather_output': 'binary_crossentropy', 'other_output': 'binary_crossentropy'},
        #loss='binary_crossentropy',
        loss_weights={'weather_output': 4./17., 'other_output': 13./17.},
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

    batch_size = 48
    epochs = 200

    checkpoint = ModelCheckpoint('RESNET50_224_bottleneck_top.model',
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True
                                )

    earlystop  = EarlyStopping(monitor='val_loss',
                               min_delta=0.0002,
                               patience=5, verbose=0, mode='auto')


    history = model.fit(
        {'main_input' : x_train},
        {'weather_output' : y_weather_train, 'other_output' : y_other_train},
        batch_size = batch_size,
        epochs=epochs,
        validation_data=(x_val,
                         {'weather_output':y_weather_val,'other_output':y_other_val}
                        ),
        #callbacks=[checkpoint, earlystop]
        callbacks=[checkpoint,]
    )

if validate:
    from sklearn.metrics import fbeta_score
    def f2_score(y_true, y_pred):
        # fbeta_scORE Throws a confusing error if inputs are not numpy arrays
        y_true, y_pred, = np.array(y_true), np.array(y_pred)
        # We need to use average='samples' here, any other average method will generate bogus results
        return fbeta_score(y_true, y_pred, beta=2, average='samples')

    w_val,o_val = model.predict(x_val)
    w_train,o_train = model.predict(x_train)
    for thres in np.arange(0.05,0.7,0.01):
      truth_val = np.concatenate( [y_weather_val, y_other_val] , axis=1)
      pred_val  = np.array(np.concatenate( [w_val, o_val] , axis=1)>thres, dtype='int')
      truth_train = np.concatenate( [y_weather_train, y_other_train] , axis=1)
      pred_train  = np.array(np.concatenate( [w_train, o_train] , axis=1)>thres, dtype='int')
      print(thres, "fin model train/val f2",f2_score(truth_train, pred_train), f2_score(truth_val, pred_val))

    model = load_model('RESNET50_224_bottleneck_top.model')
    model.compile(
        #loss={'weather_output': 'categorical_crossentropy', 'other_output': 'binary_crossentropy'},
        loss={'weather_output': 'binary_crossentropy', 'other_output': 'binary_crossentropy'},
        #loss='binary_crossentropy',
        loss_weights={'weather_output': 4./17., 'other_output': 13./17.},
        optimizer=optimizers.Adam(),
        #optimizer=optimizers.SGD(
        #    lr=1e-3,
        #    momentum=0.9),
            metrics=['accuracy']
        )

    w_val,o_val = model.predict(x_val)
    w_train,o_train = model.predict(x_train)
    for thres in np.arange(0.05,0.7,0.01):
      truth_val = np.concatenate( [y_weather_val, y_other_val] , axis=1)
      pred_val  = np.array(np.concatenate( [w_val, o_val] , axis=1)>thres, dtype='int')
      truth_train = np.concatenate( [y_weather_train, y_other_train] , axis=1)
      pred_train  = np.array(np.concatenate( [w_train, o_train] , axis=1)>thres, dtype='int')
      print(thres, "best model train/val f2",f2_score(truth_train, pred_train), f2_score(truth_val, pred_val))

    w_val,o_val = model.predict(x_val)
    w_train,o_train = model.predict(x_train)
    thres_list = np.arange(0.05,0.7,0.01)
    f2_score_train = []
    f2_score_val = []
    truth_val = np.concatenate( [y_weather_val, y_other_val] , axis=1)
    truth_train = np.concatenate( [y_weather_train, y_other_train] , axis=1)
    for thres in thres_list:
      pred_val  = np.array(np.concatenate( [w_val, o_val] , axis=1)>thres, dtype='int')
      pred_train  = np.array(np.concatenate( [w_train, o_train] , axis=1)>thres, dtype='int')
      f2_score_train.append( [f2_score(truth_train[:,idx], pred_train[:,idx]) for idx in range( truth_train.shape[1] )] )
      f2_score_val  .append( [f2_score(truth_val  [:,idx], pred_val  [:,idx]) for idx in range( truth_val.shape[1] )] )
    f2_score_train = np.array(f2_score_train)
    f2_score_val = np.array(f2_score_val)
    best_f2_score = f2_score_val.max(axis=0)
    best_thres = thres_list[ f2_score_val.argmax(axis=0) ]
    print( "best f2 score")
    print( best_f2_score)
    print( "best thres")
    print( best_thres)

    pred_val  = np.array(np.concatenate( [w_val, o_val] , axis=1))
    pred_tags_val = np.zeros( pred_val.shape, dtype='int' )
    for idx in range(17):
        pred_tags_val[:,idx] = pred_val[:,idx]>best_thres[idx]
    fin_f2_score = f2_score(truth_val, pred_tags_val)
    print( "fin f2 score", fin_f2_score)

    #print(thres, "best model train/val f2",f2_score(truth_train, pred_train), f2_score(truth_val, pred_val))
    np.save(open('RESNET50_224_bottleneck_top_best_thres.npy', 'wb'), best_thres)



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
