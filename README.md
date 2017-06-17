# Kaggle_Amazon_2017
Kaggle Amazon

# Data

Image: 4 channels (red, green, blue, and near infrared)

# Problems/Use:
- Use Better Metric to see the performance
```
wget https://raw.githubusercontent.com/fchollet/keras/master/keras/metrics.py
sudo cp metrics.py /usr/local/lib/python2.7/dist-packages/keras/
```

- GPU machines?

- explore and decide trade off in optimizing GRAM usage
  - larger batch size -> need more gram
  - larger input image size -> need more gram
  - larger model -> need more gram
  - freeze some weights -> reduce gram usage?

- freeze low level weights in pretained model?
  - https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

- use jpeg vs TIFF
  - tiff has 4 channels: RGB + near IR, better?
  - problem: most pretrained CNN model only have 3channels + tiff has large size

- how to combine models?
  - use XGBoost? example code: https://www.kaggle.com/opanichev/xgb-starter
  
# Tips:

- to use the multiple GPUs in Keras
```model = make_parallel(model, 8) #to use 8 GPU```

# References:
- Hierarchical Probabilistic Neural Network Language Model (Hierarchical Softmax)
http://cpmarkchang.logdown.com/posts/276263--hierarchical-probabilistic-neural-networks-neural-network-language-model

# Graphs:
![Image of InceptionV3](https://media.licdn.com/mpr/mpr/AAEAAQAAAAAAAAzbAAAAJGI1MzE2MDA2LTkxY2EtNDk3OC1hM2RjLWM0YTljNDIxMDQ1Zg.png)

# Meeting Summary:

## 2016-6-17

Roy present:
- Task: Label image class, can have multiple label per image
- photos: 40000 train, 60000 test
- 256x256 jpg, 10k per image
- TIFF is 20x larger in size and TIFF has 4 channels
R, G, B, NearIR

Problem:
From 3 channel model to 4 channel

## Roy's experience on Invasive Competition

- Run on InceptionV3, ResNet50, VGG.

### combine 3 models output q1, q2, q3:

find output p then minimize sum of KL divergence:
``` KL(p, q1) + KL(p, q2) + KL(p, q3) ```
Result becomes better

### Paper sharing

*Distilling the knowledge in a neural network (by Hinton, Jeff Dean)*

#### Ideas from paper

- Train general CNN model
- Make a covariance matrix of prediction categories
- Make confusion matrix
- inctrased concentration of relevant class in train samples, irrlevent class merged to dustbin class

- minimize sum of KL divergence (from generel model + expert model)

## To Do
- *biased training set*
