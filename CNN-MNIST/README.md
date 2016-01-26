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
+ Layer 1 Weights: W_conv1 is shape [5, 5, 1, 32]
+ Layer 1 Biases : b_conv1 is shape [32]
+ Layer 2 Weights: W_conv2 is shape [5, 5, 32, 64]
+ Layer 2 Biases : b_conv2 is shape [64]
Fully Connected Layer Weights
After convolutional layers, feature size has been reduced to a 7x7 feature.  We are using a FCL of 1024 neurons.
Weights are shape [feature size x * feature size y * num input channels, num output channels]
Biases are shape [num output channels]
+ Full Layer Weights: W_fcl is shape [7 * 7 * 64, 1024]
+ Full Layer Biases : b_fcl is shape [1024]

####Input Layer
4D uint numpy array [index, y, x, depth] of pixel values from 0 to 255.  Depth = 1.

####Convolutional Layer 1
+ Input image : x_image is a zero padded 4D tensor of shape [image_length, 28, 28, 1]
+ Output feature map: h_conv1 is a 4D tensor of shape [32, 28, 28, 1]
+ Filters: 32
+ Filter size: 5x5
+ Filter stride: 1

######Zero Padding
Zero padding simply means a border of zeros is added around the 2D image.  This allows the output features to have the same size as the original non-padded input image.  For example, if we have a 3x3 filter, the input image would need 2 borders of zeros to retain same size as the input image. Note, in this example I decided to draw the input image with a depth of 3.  For MNIST our depth for the images is 1 because the images are monochrome, but I decided to draw multiple channels as a general example:
```
                Input Image  
               (zero padded) 
  Input         -----------    
  Image        | -----------   
 -------       || -----------  
| -------      ||| 000000000 |  
|| -------     ||| 000000000 | 
||| 11133 |    ||| 001113300 |  
||| 31113 |    ||| 003111300 |  
||| 33113 | -> ||| 003311300 | 
 || 33113 |    ||| 003311300 | 
  | 31133 |    ||| 003113300 | 
   -------      || 000000000 | 
                 | 000000000 | 
                  -----------  
```
######Convolution
The 2D input image X is convolved with a 2D filter W, then the result is added by a bias b.  This yields a feature map, h_conv1a, for the convolutional layer.  Note: the variable 'h' is usually convention for 'feature map'.   Imagine the input image is a translucent blue window(our zero padded 28x28 image).  Now, imagine the filter is a smaller translucent red window(our 5x5 filter).  If you place the red filter window over the blue image window and shine a light through both, you get a purple area the size of the filter window.  Keeping this picture in your head we then apply a 2D convolution upon the "purple area" where each element of the blue image in that area is multiplied by its repsective element in the red filter.  Once all elements have been multiplied, they are summed and then a bias is added to the result.  This summation is stored in the first element of the feature map.  Then the red filter slides over to the next part of the blue image and the convolution process is continued until all elements of the image have been convolved and feature map is complete.  How far the filter moves across the image is usually called "stride". 

```           
Input Image                          
(zero padded)                              Conv
x_image                                    Feature Map
 -----------      Conv                     h_conv1a   
| -----------     Filter                  ---------   
|| -----------    W_conv1                | ---------  
||| 000000000 |    -----                 || --------- 
||| 000000000 |   | -----                ||| 1121100 |
||| 001110000 |   || -----               ||| 0223110 |
||| 000111000 |   ||| 101 |    Bias      ||| 1143310 |
||| 000011000 | *  || 010 | + b_conv1  = ||| 0124320 |
||| 000011000 |     | 101 |              ||| 0123310 |
||| 000110000 |      -----                || 0022110 |
 || 000000000 |                            | 0111100 |
  | 000000000 |                             --------- 
   -----------                         

h_conv1a = (X * W_conv1) + b_conv1
```
######ReLU Activation Function
Then feature map h_conv1a is then inputted into an activation function.  In this case our activation function is a rectified linear unit(ReLU) whos function looks like:
```
ReLU(h_conv1a) = max(0,h_conv1a)
```
Imagine a simple neuron with an activation function, A. The ReLU zeros the feature map if it is below the 0 threshold.  Otherwise the feature map value is passed through. 
```
                                           ------
h_conv1a = (x_image * W_conv1) + b_conv1  |      |  h_conv1 = ReLU(h_conv1a)              
----------------------------------------->| ReLU |----------------------------> 
Input Feature Map                         |      | Output Rectified Feature Map 
                                           ------

h_conv1 = ReLU((x_iamge * W_conv1) + b_conv1)  where h_conv1 = 0 if h_conv1a < 0 or h_conv1 = h_conv1a otherwise
```
####Max Pool Layer 1
+ Max pool: 2x2
+ Input feature map : A_conv1 is a 4D tensor of shape [32, 28, 28, 1]
+ Output feature map: A_pool1 is a 4D tensor of shape [32, 14, 14, 1]

The rectified feature map then undergoes a max pooling layer.  Max pooling operates just like the metaphor of the sliding red glass on the blue glass. However, instead of convolution the max pool layer takes the maximum value of the activations in a specified area from the input feature map and builds a new max pool feature map.  Note, a 2x2 max pool layer will halve the size of the feature map.  For example:
```
Conv                   
Feature Map               
(after ReLU)     Max     
h_conv1          Pool
---------        Feature Map    
| ---------      h_pool1 
|| ---------      ------   
||| 1121100 |    | ------  
||| 0223110 |    || ------ 
||| 1143310 |    ||| 2310 |
||| 0124320 | -> ||| 1430 |
||| 0123310 |     || 1330 |
 || 0022110 |      | 1110 |
  | 0111100 |       ------ 
   ---------             
```
####Convolutional Layer 2
+ Input feature map : h_pool1 is a zero padded 4D tensor of shape [32, 14, 14, 1]
+ Output feature map: h_conv2 is a 4D tensor of shape [64, 14, 14, 1]
+ Filters: 64
+ Filter size: 5x5
+ Filter stride: 1

Operates just like Layer 1.
####Max Pool Layer 2
+ Max pool: 2x2
+ Input feature map : h_conv2 is a 4D tensor of shape [64, 14, 14, 1]
+ Output feature map: h_pool2 is a 4D tensor of shape [64,  7,  7, 1]

Operates just like Max Pool Layer 1
####Fully Connected Layer
+ Input connections : h_pool2 is a 4D tensor of shape [64, 7, 7, 1] reshaped into a batch of vectors.  The vectors have shape [-1, 7 * 7 * 64]
+ Output connections: h_fcl is a 4D tensor of shape [1024]
```
                                    ------
h_fcla = (h_pool2 x W_fcl) + b_fcl |      |  h_fcl = ReLU(h_fcla)              
---------------------------------->| ReLU |----------------------------> 
Input Feature Map                  |      | Output Rectified Feature Map
                                    ------

h_fcl = ReLU((h_pool2 x W_fcl) + b_fcl)  where h_fcl = 0 if H_fcla < 0 or h_fcl = h_fcla otherwise
```
####Output Layer
