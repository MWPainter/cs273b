import math
import numpy as np
import tensorflow as tf

def weight_variable(shape, name = ""):
	return tf.get_variable(name+"_weight", shape = shape, 
		initializer=tf.contrib.layers.xavier_initializer())

def batch_norm(x, is_train):
	return tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training = is_train)


def bias_variable(shape, name = ""):
	return tf.get_variable(name+"_bias", shape = shape, 
		initializer = tf.constant_initializer(0.1))


def conv2d(x,W, strides = [1,1,1,1], padding = "SAME"):
	return tf.nn.conv2d(x,W,strides = strides, padding = padding)

def conv1d(x,W,stride, padding = "SAME"):
	return tf.nn.conv1d(x,W,stride = stride, padding = padding)


def max_pool(x, size = 2, strides = [1,1,1,1], padding = 'SAME', name = ""):
	with tf.variable_scope("max_pool%ix%i"%(size,size)):
		return tf.nn.max_pool(x, ksize =[1,size,size,1], strides = strides, padding = padding)

def max_pool1d(x, size, stride, padding = 'SAME', name = ""):
	with tf.variable_scope("max_pool%i"%size):
		return tf.nn.pool(x, [size], 'MAX', padding, strides = [stride])

def avg_pool(x, size = 2, strides = [1,1,1,1], padding = 'SAME', name = ""):
	with tf.variable_scope("avg_pool%ix%i"%(i,i)):
		return tf.nn.avg_pool(x, ksize=[1,size,size,1], strides=strides, padding = padding, name = name)


def conv2dLayer(x, filterSize, outputDim, strides = [1,1,1,1], padding = 'SAME',name = ""):
	with tf.variable_scope("conv2d"):
		shape = x.get_shape().as_list()
		inputDim = shape[3]

		W = weight_variable([filterSize[0],filterSize[1],inputDim,outputDim], name = name)
		b = bias_variable([outputDim], name = name)
		conv = conv2d(x, W, strides, padding = padding)
		return tf.nn.relu(tf.nn.bias_add(conv, b))

def conv1dLayer(x, filterSize, outputDim, stride = 1, padding = "SAME", name = ""):
	with tf.variable_scope("conv1d"):
		shape = x.get_shape().as_list()
		inputDim = shape[2]

		W = weight_variable([filterSize,inputDim,outputDim], name = name)
		b = bias_variable([outputDim], name = name)
		conv = conv1d(x, W, stride, padding = padding)
		return tf.nn.relu(tf.nn.bias_add(conv, b))



def denseLayer(x, outputDim, name = ""):
	with tf.variable_scope("dense"):
		shape = x.get_shape().as_list()
		inputDim = None
		if len(shape) == 3:
			inputDim = shape[1]*shape[2]
		if len(shape) == 4:
			inputDim = shape[1]*shape[2]*shape[3]
		if len(shape) == 2:
			inputDim = shape[1]
		reshape = [-1, inputDim]

		W = weight_variable([inputDim,outputDim], name = name)
		b = bias_variable(outputDim, name = name)
		if len(shape) == 4 or len(shape) == 3:
			x = tf.reshape(x, reshape)
		return tf.nn.relu(tf.nn.bias_add(tf.matmul(x, W), b))


def dropoutLayer(x, keep_prob, name = ""):
	with tf.variable_scope("dropout"):
		return tf.nn.dropout(x, keep_prob, name = name)


def readoutLayer(x, outputDim, name = ""):
	with tf.variable_scope("readout"):
		shape = x.get_shape().as_list()
		inputDim = shape[1]

		W = weight_variable([inputDim, outputDim], name = name)
		b = bias_variable([outputDim], name = name)
		return tf.nn.bias_add(tf.matmul(x, W),b)
