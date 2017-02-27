import numpy as np
import tensorflow as tf

import roi_pooling_op_test_util as util

NUM_CHANNELS = 2
SPATIAL_SCALE = 1.0/10

#-------------------------------------------------------------
#from roi_pooling_op import roi_pool

import os

filename = os.path.realpath(
  os.path.join(os.getcwd(), 'roi_pooling_op.so'))

_roi_pooling_module = tf.load_op_library(filename)
roi_pool = _roi_pooling_module.roi_pool
#-------------------------------------------------------------

conv_features = util.create_conv_features(num_channels=NUM_CHANNELS, format='NHWC')
print("conv_features shape: {}".format(conv_features.shape))
print("conv_features channel 0:\n {}".format(conv_features[0,:,:,0]))
print("conv_features channel 1:\n {}".format(conv_features[0,:,:,1]))

# ROIS shape: [NUM_ROIS, ROI]
# Each ROI value: [roi_batch_ind, start_w, start_h, end_w, end_h]
# ROI length = ((end - start) / spatial_scale) + 1 
roi_batch_ind = 0 # corresponds to the image / batch number 
rois = np.array(
  [[roi_batch_ind, 10, 10, 30, 30],  # 3 x 3 ROI, bin_size_h: 1
   [roi_batch_ind, 10, 10, 30, 40],  # 4 x 3 ROI, bin_size_h: 1.33333
   [roi_batch_ind, 10, 10, 30, 50],  # 5 x 3 ROI, bin_size_h: 1.66667
   [roi_batch_ind, 10, 10, 30, 60],  # 6 x 3 ROI, bin_size_h: 2,
   [roi_batch_ind, 15, 15, 35, 65],  # 6 x 3 ROI, bin_size_h: 2, not on spatial scale boundary
   [roi_batch_ind, 10, 20, 90, 70],  # 6 x 9 ROI
   [roi_batch_ind, 20, 20, 100, 70], # 6 x 9 ROI, clipped by 10 on left
   [roi_batch_ind, 30, 20, 110, 70], # 6 x 9 ROI, clipped by 20 on left
   [roi_batch_ind, 40, 20, 120, 70], # 6 x 9 ROI, clipped by 30 on left
   [roi_batch_ind, 50, 20, 130, 70], # 6 x 9 ROI, clipped by 40 on left
   [roi_batch_ind, 30, 10, 10, 30]]) # malformed ROI (roi_end_w  < roi_start_w) forced to be n x 1

conv_features_in = tf.placeholder(tf.float32, [None, 10, 10, NUM_CHANNELS]) # [N,H,W,C]
rois_in = tf.placeholder(tf.float32, [None, 5])  # [NUM_ROIS, ROI] # Each ROI [roi_batch_ind, start_w, start_h, end_w, end_h] 

y, argmax = roi_pool(conv_features_in, rois_in, 3, 3, SPATIAL_SCALE)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
  y_out, argmax_out = sess.run([y, argmax], {conv_features_in:conv_features, rois_in:rois})
  
print("y_out shape: {}".format(y_out.shape))
y_channel_0 = y_out[:,:,:,0]
y_channel_1 = y_out[:,:,:,1]
print("y_channel_0:\n{}".format(y_channel_0))
print("y channel 1:\n{}".format(y_channel_1))

print("argmax_out shape: {}".format(argmax_out.shape))
argmax_channel_0 = argmax_out[:,:,:,0]
argmax_channel_1 = argmax_out[:,:,:,1]
print("argmax_channel_0:\n{}".format(argmax_channel_0))
print("argmax channel 1:\n{}".format(argmax_channel_1))

#print("argmax_channel_0 converted indices:\n{}".format(util.convert_indices(argmax_channel_0, 0, NUM_CHANNELS)))
#print("argmax_channel_1 converted indices:\n{}".format(199 - util.convert_indices(argmax_channel_1, 1, NUM_CHANNELS)))
                
expected_y_channel_0 = np.array(
  [[[ 11.,  12.,  13.],
  [ 21.,  22.,  23.],
  [ 31.,  32.,  33.]],
 [[ 11.,  12.,  13.],
  [ 21.,  22.,  23.],
  [ 41.,  42.,  43.]],
 [[ 11.,  12.,  13.],
  [ 31.,  32.,  33.],
  [ 51.,  52.,  53.]],
 [[ 21.,  22.,  23.],
  [ 41.,  42.,  43.],
  [ 61.,  62.,  63.]],
 [[ 32.,  33.,  34.],
  [ 52.,  53.,  54.],
  [ 72.,  73.,  74.]],   
 [[ 33.,  36.,  39.],
  [ 53.,  56.,  59.],
  [ 73.,  76.,  79.]],
 [[ 34.,  37.,  39.],
  [ 54.,  57.,  59.],
  [ 74.,  77.,  79.]],
 [[ 35.,  38.,  39.],
  [ 55.,  58.,  59.],
  [ 75.,  78.,  79.]],
 [[ 36.,  39.,   0.], # An empty pooling region is indicated by zero 
  [ 56.,  59.,   0.],
  [ 76.,  79.,   0.]],
 [[ 37.,  39.,   0.],
  [ 57.,  59.,   0.],
  [ 77.,  79.,   0.]],
 [[  0.,   0.,  13.],
  [  0.,   0.,  23.],
  [  0.,   0.,  33.]]])
np.testing.assert_almost_equal(y_channel_0, expected_y_channel_0)

expected_y_channel_1 = np.array(
  [[[ 188.,  187.,  186.],
    [ 178.,  177.,  176.],
    [ 168.,  167.,  166.]],
   [[ 188.,  187.,  186.],
    [ 178.,  177.,  176.],
    [ 168.,  167.,  166.]],  
   [[ 188.,  187.,  186.],
    [ 178.,  177.,  176.],
    [ 158.,  157.,  156.]], 
   [[ 188.,  187.,  186.],
    [ 168.,  167.,  166.],
    [ 148.,  147.,  146.]],
   [[ 177.,  176.,  175.],
    [ 157.,  156.,  155.],
    [ 137.,  136.,  135.]],
   [[ 178.,  175.,  172.],
    [ 158.,  155.,  152.],
    [ 138.,  135.,  132.]], 
   [[ 177.,  174.,  171.],
    [ 157.,  154.,  151.],
    [ 137.,  134.,  131.]], 
   [[ 176.,  173.,  170.],
    [ 156.,  153.,  150.],
    [ 136.,  133.,  130.]],
   [[ 175.,  172.,    0.],
    [ 155.,  152.,    0.],
    [ 135.,  132.,    0.]],
   [[ 174.,  171.,    0.],
    [ 154.,  151.,    0.],
    [ 134.,  131.,    0.]],
   [[   0.,    0.,  186.],
    [   0.,    0.,  176.],
    [   0.,    0.,  166.]]])
np.testing.assert_almost_equal(y_channel_1, expected_y_channel_1)

expected_argmax_channel_0 = np.array(
  [[[ 22,  24,  26],
    [ 42,  44,  46],
    [ 62,  64,  66]],
   [[ 22,  24,  26],
    [ 42,  44,  46],
    [ 82,  84,  86]],
   [[ 22,  24,  26],
    [ 62,  64,  66],
    [102, 104, 106]],
   [[ 42,  44,  46],
    [ 82,  84,  86],
    [122, 124, 126]],
   [[ 64,  66,  68],
    [104, 106, 108],
    [144, 146, 148]],   
   [[ 66,  72,  78],
    [106, 112, 118],
    [146, 152, 158]],
   [[ 68,  74,  78],
    [108, 114, 118],
    [148, 154, 158]],
   [[ 70,  76,  78],
    [110, 116, 118],
    [150, 156, 158]],
   [[ 72,  78,  -1], # If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    [112, 118,  -1],
    [152, 158,  -1]],
   [[ 74,  78,  -1],
    [114, 118,  -1],
    [154, 158,  -1]],
   [[ -1,  -1,  26],
    [ -1,  -1,  46],
    [ -1,  -1,  66]]])
np.testing.assert_almost_equal(argmax_channel_0, expected_argmax_channel_0)

expected_argmax_channel_1 = np.array(
  [[[ 23,  25,  27],
    [ 43,  45,  47],
    [ 63,  65,  67]],
   [[ 23,  25,  27],
    [ 43,  45,  47],
    [ 63,  65,  67]],
   [[ 23,  25,  27],
    [ 43,  45,  47],
    [ 83,  85,  87]],
   [[ 23,  25,  27],
    [ 63,  65,  67],
    [103, 105, 107]],
   [[ 45,  47,  49],
    [ 85,  87,  89],
    [125, 127, 129]],  
   [[ 43,  49,  55],
    [ 83,  89,  95],
    [123, 129, 135]],
   [[ 45,  51,  57],
    [ 85,  91,  97],
    [125, 131, 137]],
   [[ 47,  53,  59],
    [ 87,  93,  99],
    [127, 133, 139]],
   [[ 49,  55,  -1],
    [ 89,  95,  -1],
    [129, 135,  -1]],
   [[ 51,  57,  -1],
    [ 91,  97,  -1],
    [131, 137,  -1]],
   [[ -1,  -1,  27],
    [ -1,  -1,  47],
    [ -1,  -1,  67]]]) 
np.testing.assert_almost_equal(argmax_channel_1, expected_argmax_channel_1)

print("\n----- All tests passed -----")
