import numpy as np


def create_conv_features(height=10, width=10, num_channels=3, format='NHWC'):
  patterns = np.zeros([4, height, width])
  seq = np.arange(100).astype(float)
  rseq = np.flipud(seq)
  patterns[0] = seq.reshape([height, width])
  patterns[1] = rseq.reshape([height, width])
  patterns[2] = patterns[0].transpose()
  patterns[3] = patterns[1].transpose()
  
  conv_features = np.zeros([num_channels, height, width])
  for channel in range(num_channels):
    conv_features[channel] = patterns[channel % 4] + channel * height * width
    
  if format == 'NHWC':
    conv_features = conv_features.transpose([1,2,0])

  return np.expand_dims(conv_features, axis=0)


def convert_indices(indices, channel, num_channels, format='NHWC'):
  return (indices - channel) / num_channels


if __name__ == "__main__":
  conv_features = create_conv_features(num_channels=4, format='NHWC')
  print("conv_features shape: {}".format(conv_features.shape))
  print("conv_features channel 0:\n {}".format(conv_features[0,:,:,0]))
  print("conv_features channel 1:\n {}".format(conv_features[0,:,:,1]))
  print("conv_features channel 2:\n {}".format(conv_features[0,:,:,2]))
  print("conv_features channel 3:\n {}".format(conv_features[0,:,:,3]))
  
  conv_features = create_conv_features(num_channels=4, format='NCHW')
  print("conv_features shape: {}".format(conv_features.shape))
  print("conv_features channel 0:\n {}".format(conv_features[0,0,:,:]))
  print("conv_features channel 1:\n {}".format(conv_features[0,1,:,:]))
  print("conv_features channel 2:\n {}".format(conv_features[0,2,:,:]))
  print("conv_features channel 3:\n {}".format(conv_features[0,3,:,:]))
