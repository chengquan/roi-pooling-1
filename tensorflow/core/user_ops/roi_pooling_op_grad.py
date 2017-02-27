import tensorflow as tf
from tensorflow.python.framework import ops

#-------------------------------------------------------------
#from roi_pooling_op import roi_pool_grad

import os

filename = os.path.realpath(
  os.path.join(os.getcwd(), 'roi_pooling_op.so'))

_roi_pooling_module = tf.load_op_library(filename)
roi_pool_grad = _roi_pooling_module.roi_pool_grad
#-------------------------------------------------------------

@ops.RegisterGradient("RoiPool")
def _roi_pool_grad(op, grad, _):
  """The gradients for `roi_pool`.
  Args:
    op: The `roi_pool` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `roi_pool` op.
  Returns:
    Gradients with respect to the input of `zero_out`.
  """
  data = op.inputs[0]
  rois = op.inputs[1]
  argmax = op.outputs[1]
  pooled_height = op.get_attr('pooled_height')
  pooled_width = op.get_attr('pooled_width')
  spatial_scale = op.get_attr('spatial_scale')

  # compute gradient
  data_grad = roi_pool_grad(data, rois, argmax, grad, pooled_height, pooled_width, spatial_scale)

  return [data_grad, None]  # List of one Tensor, since we have one input
