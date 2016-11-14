import tensorflow as tf
import numpy as np

def score_reach(input_array, nb, threshold):
    score=0
    index = 0
    while(input_array[index]<threshold):
      score = score +1
    if (score == nb):
        return True
    else:
        return False

def score_regression(prediction,target,error):
    score = 0
    index = 0
    while(index<np.array(prediction).shape[0]):
        if target[index]>0:
            if (prediction[index]>target[index]*(1-error))and(prediction[index]<target[index]*(1+error)):
                score = score +1
        else:
            if (prediction[index]<error):
                score = score +1
        index = index +1
    return 100*score/np.array(prediction).shape[0]
def bad(prediction,target,error):
    bad = []
    index = 0
    while(index<np.array(prediction).shape[0]):
        if (prediction[index]<target[index]*(1-error))or(prediction[index]>target[index]*(1+error)):
            bad = np.append(bad,[index])
            bad = np.append(bad,[prediction[index]])
            bad = np.append(bad,[target[index]])
        index = index +1
    return bad[:9]

def batch(input_array, index, batch_size): #split the inputs in several batch
    d= (index+1)*batch_size % np.array(input_array).shape[0]
    return input_array[d-batch_size:d]

def weight_variable(shape): #initialization of the Weight arrays
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape): #initialization of the bias
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W): # convolution - according to tensorflow tutorial
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x): # maxpool - according to tensorflow tutorial
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')