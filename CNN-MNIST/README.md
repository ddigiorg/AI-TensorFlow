#Convolutional Neural Network - MNIST Dataset

The purpose of this Convolutional Neural Network(CNN) project is to understand deep neural networks, familiarize myself with Machine Learning code development, and provide a code base for research on  Artificial Intelligence(AI) concepts.  My motivation is to introduce myself to the field of AI via a well-known, relatively simple, and interesting problem: handwritten digit recognition using the MNIST dataset.  The project's code is implemented by following TensorFlow's deep convolutional network [tutorial](https://www.tensorflow.org/versions/master/tutorials/mnist/pros/index.html).

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
 -------      -------      -------      -------      -------      -----------      ---------
| Input |    | Conv  |    | Max   |    | Conv  |    | Max   |    | Fully     |    | Output  |
| Layer | -> | Layer | -> | Pool  | -> | Layer | -> | Pool  | -> | Connected | -> | Softmax |
|       |    | 1     |    | 1     |    | 2     |    | 2     |    | Layer     |    | Layer   |
 -------      -------      -------      -------      -------      -----------      ---------
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

######Zero Padding
Zero padding simply means a border of zeros is added around the 2D image.  This allows the output features to have the same size as the original non-padded input image.  For example, if we have a 3x3 filter, the input image would need 2 borders of zeros to retain same size as the input image. Note, in this example I decided to draw the input image with a depth of 3.  For MNIST our depth for the images is 1 because the images are monochrome, but I decided to draw multiple channels as a general example:
```
              Input Image  
             (zero padded) 
 Input        ---------    
 Image       | ---------   
 -----       || ---------  
| -----      |||000000000| 
|| -----     |||000000000| 
|||11133|    |||001113300| 
|||31113|    |||003111300| 
|||33113| -> |||003311300| 
 ||33113|    |||003311300| 
  |31133|    |||003113300| 
   -----      ||000000000| 
               |000000000| 
                ---------  
```
######Convolution
The 2D input image X is convolved with a 2D filter W, then the result is added by a bias B.  This yields a feature map for the convolutional layer.  Imagine the input image is a translucent blue window(our zero padded 28x28 image).  Now, imagine the filter is a smaller translucent red window(our 5x5 filter).  If you place the red filter window over the blue image window and shine a light through both, you get a purple area the size of the filter window.  Keeping this picture in your head we then apply a 2D convolution upon the "purple area" where each element of the blue image in that area is multiplied by its repsective element in the red filter.  Once all elements have been multiplied, they are summed and then a bias is added to the result.  This summation is stored in the first element of the feature map.  Then the red filter slides over to the next part of the blue image and the convolution process is continued until all elements of the image have been convolved and feature map is complete.  How far the filter moves across the image is usually called "stride". 

```           
Input Image                          
(zero padded)                  
     X                          Conv
---------       Conv            Feature   
| ---------     Filter          -------   
|| ---------      W            | -------  
|||000000000|    ---           || ------- 
|||000000000|   | ---          |||1121100|
|||001110000|   || ---         |||0223110|
|||000111000|   |||101|  Bias  |||1143310|
|||000011000| *  ||010| + B  = |||0124320|
|||000011000|     |101|        |||0123310|
|||000110000|      ---          ||0022110|
 ||000000000|                    |0111100|
  |000000000|                     ------- 
   ---------                         
```
```
feature = (X * W) + B
```
######ReLU Activation Function
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
+ Max pool: 2x2
+ Input feature: 32 28x28 features from Convolutional Layer 1
+ Output feature: 32 14x14 features

The rectified feature map then undergoes a max pooling layer.  Max pooling operates just like the metaphor of the sliding red glass on the blue glass. However, instead of convolution the max pool layer takes the maximum value of the activations in a specified area from the input feature and builds a new max pool feature.  Note, a 2x2 max pool layer will halve the size of the feature.  For example:
```
Conv                   
Feature                
(after ReLU)   Max     
 -------       Pool    
| -------      Feature 
|| -------      ----   
|||1121100|    | ----  
|||0223110|    || ---- 
|||1143310|    |||2310|
|||0124320| -> |||1430|
|||0123310|     ||1330|
 ||0022110|      |1110|
  |0111100|       ---- 
   -------             
```
####Convolutional Layer 2
+ Input channels: 32 zero padded 14x14 features outputted by Max Pool Layer 1
+ Filters: 64
+ Filter size: 5x5
+ Filter stride: 1
+ Output channels: 64 features

Operates just like Layer 1.
####Max Pool Layer 2
+ Max pool: 2x2
+ Input feature: 64 14x14 features from Convolutional Layer 2 
+ Output feature: 64 7x7 features

Same procedure as Max Pool Layer 1
####Fully Connected Layer

####Output Layer
