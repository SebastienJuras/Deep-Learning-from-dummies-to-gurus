import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt
import tensorflow as tf
import random as rd

# variables
nb=100
size=14
image_flat = np.zeros((nb,size*size),dtype=np.float32)
label = np.zeros((nb,1),dtype=np.float32)
tcost=[]
tb=[]
twmean=[]
twstd=[]
ti=[]

# creation of the inputs and the labels
index = 0
for d in range(nb):
    image = np.array([rd.random() for k in range(size*size)])
    image_flat[index] = image.flatten()
    label[index] = [0.5*np.sum(image_flat[index])+1]
    index = index + 1

# functions

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

#use tensorflow and try to find W & B where  Y = W * X + b is closed to Label
sess = tf.InteractiveSession()

with tf.name_scope("input") as scope:
    x = tf.placeholder(tf.float32, [None, size*size],name="x-input")
    yt = tf.placeholder(tf.float32, [None, 1],name="y-input")

with tf.name_scope("layer1") as scope:
    W_conv1 = weight_variable([5, 5, 1, 1])
    b_conv1 = bias_variable([1])
    x_image = tf.reshape(x, [-1, size, size, 1])
    conv = conv2d(x_image, W_conv1)
    h_conv1 = tf.nn.relu(conv+b_conv1,name="out1")
    yp = tf.reduce_sum(h_conv1)

with tf.name_scope("optimization") as scope:
    cost = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(yt, yp))))
    train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)

sess.run(tf.initialize_all_variables())
ct_train = int(nb*0.7)

for i in range(100000):
  Xin = batch(image_flat[:ct_train],i,1)
  Xout = batch(label[:ct_train],i,1)
  train_step.run(feed_dict={x: Xin, yt: Xout})
  if (i % 100 ==0):
    z = yp.eval(feed_dict={x: Xin, yt: Xout})
    z2 = h_conv1.eval(feed_dict={x: Xin, yt: Xout})
    c = cost.eval(feed_dict={x: Xin, yt: Xout})
    convo = conv.eval(feed_dict={x: Xin, yt: Xout})
    print "cost",i,c,b_conv1.eval(),np.array(convo).mean(),np.array(convo).std()
    tcost=np.append(tcost,[c])
    tb = np.append(tb, [b_conv1.eval()])
    twmean = np.append(twmean, [np.array(convo).mean()])
    twstd = np.append(twstd, [np.array(convo).std()])
    ti = np.append(ti, [i])
print "fin"
plt.subplot(411)
plt.plot(tcost)
plt.title('cost')
plt.subplot(412)
plt.plot(tb)
plt.title('bias')
plt.subplot(413)
plt.plot(twmean)
plt.title('weight mean')
plt.subplot(414)
plt.plot(twstd)
plt.title('weight std')
plt.show()