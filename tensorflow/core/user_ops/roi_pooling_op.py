import tensorflow as tf
import os

filename = os.path.realpath(
  os.path.join(os.getcwd(), 'roi_pooling_op.so'))

_roi_pooling_module = tf.load_op_library(filename)
roi_pool = _roi_pooling_module.roi_pool
