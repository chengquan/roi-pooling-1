import tensorflow as tf
import numpy as np
import roi_pooling_op_grad 


#-------------------------------------------------------------
#from roi_pooling_op import roi_pool

import os

filename = os.path.realpath(
  os.path.join(os.getcwd(), 'roi_pooling_op.so'))

_roi_pooling_module = tf.load_op_library(filename)
roi_pool = _roi_pooling_module.roi_pool
#-------------------------------------------------------------


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

array = np.random.rand(32, 100, 100, 3)
data = tf.convert_to_tensor(array, dtype=tf.float32)
rois = tf.convert_to_tensor([[0, 10, 10, 20, 20], [31, 30, 30, 40, 40]], dtype=tf.float32)

W = weight_variable([3, 3, 3, 1])
h = conv2d(data, W)

[y, argmax] = roi_pool(h, rois, 6, 6, 1.0/3)
y_data = tf.convert_to_tensor(np.ones((2, 6, 6, 1)), dtype=tf.float32)
print y_data, y, argmax

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

# Launch the graph.
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
  sess.run(init)
  for step in xrange(10):
    sess.run(train)
    print("Step: {}".format(step))
    print("Weights:\n{}".format(sess.run(W)))
    print("roi_pool output:\n{}".format(sess.run(y)))
