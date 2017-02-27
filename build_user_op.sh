if [ -n "$CUDA_HOME" ]; then
    bazel build -c opt --config=cuda //tensorflow/core/user_ops:roi_pooling_op.so
else
	bazel build -c opt //tensorflow/core/user_ops:roi_pooling_op.so
fi
