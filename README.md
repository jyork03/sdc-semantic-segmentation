# Semantic Segmentation
### Introduction
In this project, Fully Convolutional Network (FCN), using a modified and pre-trained `VGG16` network as an encoder, is used
to classify the pixels of road images into either road/notroad classes.  The road-labeled pixels are color-coded in 
green on the result images in the `./runs` folder.

### Architecture
The model used for this project is based on the 
[FCN-8 architecture](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) created at Berkeley.

First, the program loads the pre-trained `VGG16` model for use as a FCN encoder in the `load_vgg()` function.  Next, it
initializes the architecture for the decoder in `layers()`.  The decoder model is made up of a series of upsampling
layers, using transposed convolutions, and skip connections, connecting tensors from the vgg model to layers in the 
decoder with 1x1 convolutions.

Next, it's trained with an adam optimizer to minimize cross-entropy loss.  I took an iterative approach to hyper
parameter tuning, and eventually settled on a learning rate of 1e-4, a dropout keep probability of 0.5, a batch size of 
16 and found that running it for around 20 epochs produced reasonably good results.

### Data/Image Augmentation
The dataset is relatively small, so image augmentation is important to help it generalize better to new data.  While
training, each image has chance to be augmented.  The augmentations used in this program are horizontal flipping,
brightening, darkening, and rotating randomly between -15 and 15 degrees.

### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
 - [MoviePy](https://zulko.github.io/moviepy/install.html)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Run
Run the following command to initiate the training, and generate new inference images and a video:

```
python main.py
```

This program provides a couple command line arguments for convenience.  Run `python main.py -h` for more details.

 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip).
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [post](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/forum_archive/Semantic_Segmentation_advice.pdf) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
