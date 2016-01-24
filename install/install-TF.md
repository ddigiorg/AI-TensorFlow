#Installing TensorFlow(GPU version) on Arch Linux
2016-01-24

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
Currently TensorFlow requires CUDA 7.0, an older version.  Download cuda-7.0 from the Arch User Repository(AUR):
```
$ sudo yaourt -S cuda-7.0-compat
```
Arch will install to the directory ~/opt/cuda-7.0/.  

