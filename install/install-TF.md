#Installing TensorFlow
2016-01-24
- System: Arch Linux
- Python: 3.5.2
- TensorFlow: GPU version

##1. Install Nvidia drivers and other dependancies:
```
$ pacman -S nvidia python-numpy swig
```
##2. Install CUDA Toolkit 7.0
Verify that the GPU is CUDA capable:
```
$ lspci | grep -i nvidia
```
Verify you have GCC installed:
```
$ gcc --version
```
Currently TensorFlow requires CUDA 7.0, an older version of the toolkit.  Download [cuda-7.0](https://aur.archlinux.org/packages/cuda-7.0-compat/) from the Arch User Repository(AUR):
```
$ sudo yaourt -S cuda-7.0-compat
```
Arch will install to the directory ~/opt/cuda-7.0/  
##3. Install cuDNN v2
Navigate into the CUDA Toolkit directory:
```
$ cd ~/opt/cuda-7.0/
```
Currently TensorFlow requires [cuDNN v2](https://developer.nvidia.com/rdp/cudnn-archive), and older version of the Nvidia deep learning libraries.  The bad news is you need to register for their Accelerated Computing Developer Program... (SIGHDUCK).  The good news is it is free and only requires you wait ~2 days while you contemplate your navel to get access to the download.  In the mean time you may install the CPU version of TensorFlow to pass the time or go outside! (hisss).  Anyway you need to log into their website to download the cuDNN v2 library for Linux, then move the tarball to your CUDA toolkit directory (~/opt/cuda-7.0).  Extract the tarball inside your CUDA toolkit directory:
```
$ tar -xvzf cudnn-6.5-linux-x64-v2.tgz
```
After you have extracted the tarball file it will create a cudnn-6.5-linux-x64-v2 folder.  Copy the cuDNN files into the necessary CUDA Toolkit directories:
```
$ sudo cp cudnn-6.5-linux-x64-v2/cudnn.h /opt/cuda-7.0/include
$ sudo cp cudnn-6.5-linux-x64-v2/libcudnn* /opt/cuda-7.0/lib64
```
Then change the cuDNN file permissions:
```
$ sudo chmod a+r /opt/cuda-7.0/lib64/libcudnn*
```
(Optional) Remove the cuDNN tarball:
```
$ rm cudnn-6.5-linux-x64-v2.tgz
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
Download the TensorFlow wheel file:
```
$ wget https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.6.0-cp34-none-linux_x86_64.whl
```
Currently this file of TensorFlow only supports Python 3.4, therefore we need to trick the installer.  Rename Tensorflow wheel file to make it compatible with Python 3.5:
```
$ mv tensorflow-0.6.0-cp34-none-linux_x86_64.whl tensorflow-0.6.0-cp35-none-linux_x86_64.whl
```
Pip install from Tensorflow wheel file: 
```
$ sudo pip3 install tensorflow-0.6.0-cp35-none-linux_x86_64.whl
```
##5. Set Environment Variables
Set the LD_LIBRARY_PATH and CUDA_HOME environment variables by adding the commands below to your user and/or root .bashrc files. These assume CUDA installation is in /opt/cuda-7.0:
```
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/cuda-7.0/lib64"
export CUDA_HOME=“/opt/cuda-7.0”
```
##6. Test TensorFlow
In the terminal type:
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
