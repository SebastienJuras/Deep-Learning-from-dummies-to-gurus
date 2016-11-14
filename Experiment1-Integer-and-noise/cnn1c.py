import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import random as rd
import cnn_fct as fct

# variables
nb_epoch=1000                                                   # nb of iteration
batch_size = 1                                                  # size of the batch of each iteration
nb=1000                                                          # nb of data
size=14                                                          # size of the data
image_flat = np.zeros((nb,size*size),dtype=np.float32)           # data flat
label = np.zeros((nb,1),dtype=np.float32)                        # label
tcost_train=[]                                                   # training cost tab
tcost_test=[]                                                    # training cost t
tb=[]
tw1=[]
tw2=[]
ct_train = int(nb*0.7)                                           # nb of training data (70% of total number of data)
ct_test = nb-ct_train                                            # nb of test data
learning_rate= 0.0001                                            # learning rate of the gradiant descent
X_range = 10                                                     # dispersion of the output and input
X_noise = 0.1                                                    # noise of the inputs

# creation of the inputs and the labels
index = 0
for d in range(nb):
    lab = rd.randint(0,X_range)
    image = np.array([X_noise*rd.random()+lab for k in range(size*size)])
    image_flat[index] = image.flatten()
    label[index] = [lab]
    index = index + 1

#use tensorflow and try to find W & B where  Y = W * X + b is closed to Label
sess = tf.InteractiveSession()

with tf.name_scope("input") as scope:
    x = tf.placeholder(tf.float32, [None, size*size],name="x-input")
    yt = tf.placeholder(tf.float32, [None, 1],name="y-input")

with tf.name_scope("layer1") as scope:
    W_conv1 = fct.weight_variable([5, 5, 1, 1])
    W2 = fct.weight_variable([size,size,1])
    b_conv1 = fct.bias_variable([1])
    x_image = tf.reshape(x, [-1, size, size, 1])
    conv = fct.conv2d(x_image, W_conv1)
    h_conv1 = tf.nn.relu(conv+b_conv1,name="out1")
    yp = tf.reduce_sum(tf.reduce_sum(h_conv1*W2,1),1)


with tf.name_scope("optimization") as scope:
    cost = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(yt, yp))))
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# run of iteration

sess.run(tf.initialize_all_variables())
Xtest = image_flat[ct_train:nb]
Ytest = label[ct_train:nb]
rw1=rd.randint(0,24)
rw2=rd.randint(0,size*size-1)

for i in range(nb_epoch):
  # prepare the batch
  X = fct.batch(image_flat[:ct_train],i,batch_size)
  Y = fct.batch(label[:ct_train],i,batch_size)
  # train the data
  train_step.run(feed_dict={x: X, yt: Y})
  # evaluate some variables and fill tab for plot
  if (i % 10 ==0):
    c = cost.eval(feed_dict={x: X, yt: Y})
    ct = cost.eval(feed_dict={x: Xtest, yt: Ytest})
    pred = yp.eval(feed_dict={x: Xtest, yt: Ytest})
    convo = conv.eval(feed_dict={x: X, yt: Y})
    print "iteration",i,"training cost",c,"test cost",ct,"score",fct.score_regression(pred,Ytest,0.5)
    #print fct.bad(pred,Ytest,0.5)
    #print pred[:10]
    #print Ytest[:10]
    tcost_train=np.append(tcost_train,[c])
    tcost_test = np.append(tcost_test, [ct])
    tb = np.append(tb, [b_conv1.eval()])
    eW1=W_conv1.eval().flatten()
    eW2 = W2.eval().flatten()
    tw1 = np.append(tw1, [eW1[rw1]])
    tw2 = np.append(tw2, [eW2[rw2]])

# display plots

plt.subplot(411)
plt.plot(tcost_train)
plt.title('training cost')
plt.ylim(0, 2)
plt.subplot(412)
plt.plot(tcost_test)
plt.title('test cost')
plt.ylim(-1, 1)
plt.subplot(413)
plt.plot(tw1)
plt.title('W1')
plt.ylim(-0.1, 0.1)
plt.subplot(414)
plt.plot(tw2)
plt.title('W2')
plt.ylim(-0.1, 0.1)
plt.show()
