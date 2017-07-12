
# coding: utf-8

import math
import numpy as np
import pandas as pd
import cv2

imglist_df = pd.read_csv("../input/train_v2.csv")
img_path = "../input/train-jpg/"

#imglist_df = pd.read_csv("../input/sample_submission_v2.csv")
#img_path = "../input/test-jpg-additional/"

#imglist_df = imglist_df[:500]

nFeatures = 2048
#train = True
train = False
predict = True

from keras import applications
from keras import optimizers

img_rows, img_cols, img_channel = 224, 224, 3
base_model = applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(
        img_rows,
        img_cols,
        img_channel))

   #model = Model(inputs=base_model.input,
   #              outputs= [
   #                        #add_weather_model(base_model.output),
   #                        #add_other_model(base_model.output),
   #                        weather_output,
   #                        other_output,
   #                       ]
   #             )

base_model.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.SGD()
       )
base_model.summary()

batch_size = 32
nbatches = math.ceil( len(imglist_df)/batch_size )
threshold = 0.5
out = np.zeros([len(imglist_df), nFeatures], dtype='float32')
x   = np.zeros( [batch_size, 224, 224, 3], dtype='float32')
batch_id = 0
while (batch_id*batch_size) < len(imglist_df):
    for i in range(batch_size):
        pos = batch_id*batch_size+i
        name = imglist_df.image_name[pos]
        # read image
        img = img_path + name + ".jpg"
        img = cv2.imread(img)

        # process img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=(224,224))
        x[i] = img / 255.

        if (pos+1)>=len(imglist_df): break
    batch_out = base_model.predict(x).reshape((batch_size,-1))
    out[batch_id*batch_size:  batch_id*batch_size+i+1] = batch_out[0:i+1]
    print ("Batch %s out of  %s done" % (batch_id, nbatches))
    batch_id+=1

np.save(open('resnet50_features_train_jpg_224.npy', 'wb'), out)
