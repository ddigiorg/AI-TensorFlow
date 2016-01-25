"""
Parameters:
	-dataset: Input "training" or "testing" to pull data from the respective MNIST data sets.
	-start: Input the first set of data you wish to pull data from.
	-stop: Input the last set of data you wish to pull data from.
	-path: Input a strong denoting the file path to the MNIST data

Returns:
	-images: a list consisting of a 2D list(OR TUPLE?) of floats denoting monochrome pixel values.
	-labels: list of unsigned integers denoting image labels from 0 to 9.

"""
import struct
import array as ar
import numpy as np

def load_mnist(dataset="training", start=1, stop=60000, path=""):

	if path:
		if dataset == "training":
			fname_lbl = path + "train-labels.idx1-ubyte"
			fname_img = path + "train-images.idx3-ubyte"
		elif dataset == "testing":
			fname_lbl = path + "t10k-labels.idx1-ubyte"  
			fname_img = path + "t10k-images.idx3-ubyte"      
		else:
			raise ValueError("dataset must be 'testing' or 'training'")

		with open(fname_lbl, "rb") as flbl:
			magic_num, size = struct.unpack( ">II", flbl.read(8) )
			lbl = ar.array( "b", flbl.read() )

		with open(fname_img, "rb") as fimg:
			magic_num, size, rows, cols = struct.unpack( ">IIII", fimg.read(16) )
			img = ar.array( "B", fimg.read() )

		size = stop - start + 1
    
		images = np.zeros( (size, rows, cols) )
		labels = np.zeros( (size, 1) )

		for i in range(size-1):
			idx = start + i - 1
			images[i] = np.array(img[ idx*rows*cols : (idx+1)*rows*cols ]).reshape((rows, cols))
			labels[i] = lbl[idx]

		return images, labels

"""
Notes:

struct.unpack(format, string)

    + This module performs conversions between Python values and C structs
      represented as Python strings.  Basically, reads the MNIST header info.
    + format: 
        + '>' is high-endian
        + 'I' is unsigned integer (4 bytes)

array.array(typecode, data)

    + puts the MNIST data into an array, which are like lists except the
      type of objects stored in them is constrained.
    + typecode:
        + 'b' is signed char (1 byte)
        + 'B' is signed char (1 byte)

"""
