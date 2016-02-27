#Installing TensorFlow
2016-02-27
- System: Arch Linux
- Python: 3.5.1
- CUDA Toolkit: 7.5.18-1
- cuDNN: 7.0 v4.0
- TensorFlow: 7.1 GPU

##1. Install Nvidia drivers and other dependancies:
I found the reboot is necessary after installing the nvidia drivers.  Otherwise I got errors launching a TF session.
```
$ pacman -S nvidia python-numpy swig
$ reboot
```
##2. Install CUDA Toolkit
Verify that the GPU is CUDA capable:
```
$ lspci | grep -i nvidia
```
Verify you have GCC installed:
```
$ gcc --version
```
Currently TensorFlow requires CUDA 7.0 or higher.
([CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit))([CUDA Toolkit Documentation](http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/#axzz3xQuiXyUB))([cuda-7.5.18-1](https://www.archlinux.org/packages/community/x86_64/cuda/))
Download and Install CUDA Toolkit from the Arch Linux repository:
```
$ pacman -S cuda
```
Arch will install to the directory ~/opt/cuda/  
##3. Install cuDNN
Navigate into the CUDA Toolkit directory:
```
$ cd ~/opt/cuda/
```
Currently TensorFlow requires cuDNN v2 or above.  The bad news is you need to register for their Accelerated Computing Developer Program... (SIGHDUCK).  The good news is it is free and only requires you wait ~2 days while you contemplate your navel to get access to the download.  In the mean time you may install the CPU version of TensorFlow to pass the time (or go outside and chase girls!).  Anyway you need to log into their website to download the cuDNN library for Linux, then move the tarball to your CUDA toolkit directory (~/opt/cuda).  Extract the tarball inside your CUDA toolkit directory:
```
$ tar -xvzf cudnn-7.0-linux-x64-v4.0-prod.tgz
```
After you have extracted the tarball file it will create a /opt/cuda/cuda folder.  Copy the cuDNN files into the necessary CUDA Toolkit directories:
```
$ sudo cp cuda/include/cudnn.h /opt/cuda/include
$ sudo cp cuda/lib64/libcudnn* /opt/cuda/lib64
```
Then change the cuDNN file permissions:
```
$ sudo chmod a+r /opt/cuda/lib64/libcudnn*
```
(Optional) Remove the cuDNN tarball:
```
$ rm cudnn-7.0-linux-x64-v4.0-prod.tgz
```
##4. Install TensorFlow(GPU version)
[TensorFlow](https://www.tensorflow.org/)

Verify pip3 is installed:
```
$ pip3 --version
```
Navigate to your home directory
```
$ cd $HOME
```
Download the TensorFlow 7.1 (GPU)  wheel file:
```
$ wget https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.7.1-cp34-none-linux_x86_64.whl

```
Currently this file of TensorFlow only supports Python 3.4, therefore we need to trick the installer.  Rename Tensorflow wheel file to make it compatible with Python 3.5:
```
$ mv tensorflow-0.7.1-cp34-none-linux_x86_64.whl tensorflow-0.7.1-cp35-none-linux_x86_64.whl
```
Pip install from Tensorflow wheel file: 
```
$ sudo pip3 install tensorflow-0.6.0-cp35-none-linux_x86_64.whl
```
##5. Set Environment Variables
Set the LD_LIBRARY_PATH and CUDA_HOME environment variables by adding the commands below to your user and/or root .bashrc files. These assume CUDA installation is in /opt/cuda/:
```
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/cuda/lib64"
export CUDA_HOME=/opt/cuda/
```
##6. Test TensorFlow
In the terminal type the following code.  Interestingly after importing tensorflow as root it was unable to load cuDNN.  However, running it as a user worked.  I think this may be because I installed TF in my user directory and root wasnt pulling from there?  both root and user .bashrc files are exactly the same... (I'm an arch linux noob so please excuse me if Im being dumb).  Anyway running as user works:
```
$ python
...
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))
Hello, TensorFlow!
>>> a = tf.constant(10)
>>> b = tf.constant(32)
>>> print(sess.run(a + b))
42
>>>
```

#Arch Linux Dependancies
##Installing wget
```
$ pacman -S wget
```
##Installing pip
```
$ pacman -S python-pip
$ pip install wheel
```
