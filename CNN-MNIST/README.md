#Convolutional Neural Network - MNIST Dataset

The purpose of this Convolutional Neural Network(CNN) project is to understand deep neural networks, familiarize myself with Machine Learning code development, and provide a code base for research into novel Artificial Intelligence(AI) concepts.  My motivation is to introduce myself to the field of AI via a well-known, relatively simple, and interesting problem: handwritten digit recognition using the MNIST dataset.  The project's code is implemented from the deep convolutional MNIST classifier from TensorFlow's Deep MNIST for Experts [tutorial](https://www.tensorflow.org/versions/master/tutorials/mnist/pros/index.html).

##MNIST Dataset
The MNIST dataset consists of images and labels of handwritten digits from 0 to 9.  The images are 28x28 pixels of monochrome values from 0 to 255.  The files are:

+ t10k-images.idx3-ubyte
+ t10k-labels.idx1-ubyte
+ train-images.idx3-ubyte
+ train-labels.idx1-ubyte

TensorFlow has created a script which downloads and installs the MNIST dataset.  Additionally, it provides a class to store the training, validation, and testing data sets as NumPy arrays.  Additionally I have developed my own script load_mnist.py, which is used to access the MNIST testing and training datasets from the MNIST files.  I've decided to use TensorFlow's script because it automates the MNIST dataset download process. 

+ images: a 4D uint numpy array [index, y, x, depth] of pixel values from 0 to 255.  Because the MNIST dataset is monochrome the depth is 1.  For reference, RGB images have a depth of 3 and CMYK would be 4.
+ labels: a 1D uint numpy array [index] of labels from 0 to 9 in onehot format (aka 4 is 000001000, 9 is 100000000, etc.).

##Convolutional Neural Network
```                                                                                                     
 Input Image                 Conv                                                          
(zero padded)               Feature                                                        
 ---------               (after ReLU)               Fully      Fully  
| ---------      Conv      -------       Max Pool   Conn       Conn   
|| ---------    Filter    | -------      Feature   Weights(W) Feature
|||000000000|    ---      || -------      ----        [w0]    [DF0xW] 
|||000000000|   | ---     |||1121100|    | ----       [w1]    [DF1xW] 
|||001110000|   || ---    |||0223110|    || ----      [w2]    [DF2xW] 
|||000111000|   |||101|   |||1143310|    |||2310|     [w3]    [DF3xW] 
|||000011000| *  ||010| = |||0124320| -> |||1430|  -> [w4] -> [DF4xW] 
|||000011000|     |101|   |||0123310|     ||1330|     [w5]    [DF5xW] 
|||000110000|      ---     ||0022110|      |1110|     [w6]    [DF6xW] 
 ||000000000|               |0111100|       ----      [w7]    [DF7xW] 
  |000000000|                -------                  [w8]    [DF8xW] 
   ---------                                          ....    ....... 
                                                                                                  
|-----------|   |-------------------|    |------|      |-------------| 
 Input Layer        Conv Layer 1        Max Pool 1     Fully Connected 
```
####Weight Initialization
+ Weights: Initialized by a normal distribution with a standard deviation of 0.1.
+ Biases: Initialized to 0.1
Convolutional Weights
Weights are shape [filter size x, filter size y, num input channels, num output channels]
Biases are shape [num output channels]
+ Layer 1 Weights: [5, 5, 1, 32]
+ Layer 1 Biases:  [32]
+ Layer 2 Weights: [5, 5, 32, 64]
+ Layer 2 Biases:  [64]
Fully Connected Layer Weights
After convolutional layers, feature size has been reduced to a 7x7 feature.  We are using a FCL of 1024 neurons.
Weights are shape [feature size x * feature size y * num input channels, num output channels]
Biases are shape [num output channels]
+ Full Layer Weights: [7 * 7 * 64, 1024]
+ Full Layer Biases: [1024]

####Input Layer
4D uint numpy array [index, y, x, depth] of pixel values from 0 to 255.  Depth = 1.

####Convolutional Layer 1
+ Input channels: 1 zero padded 28x28 image
+ Filters: 32
+ Filter size: 5x5
+ Filter stride: 1
+ Output channels: 32 features

Zero padded means just adds zeros around the border of the 2D image.  This allows the output features to have the same size as the original non-padded input image.

The 2D input image X is convolved with a 2D filter W, then the result is added by a bias B.  This yields a feature map for the convolutional layer.  Imagine the input image is a translucent blue window(our zero padded 28x28 image).  Now, imagine the filter is a smaller translucent red window(our 5x5 filter).  If you place the red filter window over the blue image window and shine a light through both, you get a purple area the size of the filter window.  Keeping this picture in your head we then apply a 2D convolution upon the "purple area" where each element of the blue image in that area is multiplied by its repsective element in the red filter.  Once all elements have been multiplied, they are summed and then a bias is added to the result.  This summation is stored in the first element of the feature map.  Then the red filter slides over to the next part of the blue image and the convolution process is continued until all elements of the image have been convolved and feature map is complete.  How far the filter moves across the image is usually called "stride". 
```
feature = (X * W) + B
```
Then feature is then inputted into an activation function.  In this case our activation function is a rectified linear unit(ReLU) whos function looks like:
```
ReLU(feature) = max(0,feature)
```
Imagine a simple neuron with an activation function, A. The ReLU zeros the feature if it is below the 0 threshold.  Otherwise the feature value is passed through. 
```
               ------
 (X * W) + B  |      |  A = ReLU((X * W) + B)        | 0 if ((X * W) + B)  < 0 
--------------| ReLU |-------------------------  A = |
Input Feature |      | Output Rectified Feature      | (X * W) + B otherwise
               ------

```
####Max Pool Layer 1
+ Max pool size: 2x2

The rectified feature map then undergoes a max pooling.


####Convolutional Layer 2

####Max Pool Layer 2

####Fully Connected Layer

####Output Layer
