# Super Resolution Of Image

enhance the resolution of the image by two.

# Runtime Environment

- python version: 3.7
- tensorflow version: 1.14.0
- opencv version 3.X
- numpy version: 1.16.5

# result

original:

![](images/small.png)

enhance:

![](images/enhance.png)

# File

- images\_to\_tfrecord.py: turn all the images into tfrecord
- configs.py: define some parameter of model 
- layer.py: define function of convolution, relu, deconcolutoion and etc.
- model.py: define model
- super\_esolution.py: test image
- train.py: train model

# Licence

MIT
