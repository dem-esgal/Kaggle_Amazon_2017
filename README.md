# Kaggle_Amazon_2017
Kaggle Amazon


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
  


# References:
- Hierarchical Probabilistic Neural Network Language Model (Hierarchical Softmax)
http://cpmarkchang.logdown.com/posts/276263--hierarchical-probabilistic-neural-networks-neural-network-language-model

# Graphs:
![Image of InceptionV3](https://media.licdn.com/mpr/mpr/AAEAAQAAAAAAAAzbAAAAJGI1MzE2MDA2LTkxY2EtNDk3OC1hM2RjLWM0YTljNDIxMDQ1Zg.png)
