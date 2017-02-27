# ROI Pooling

A TensorFlow/C++ implementation of the ROI Pooling operation from "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks" (https://arxiv.org/abs/1506.01497).    

TensorFlow lacks a Region-of-Interest (ROI) pooling operation required to implement Faster R-CNN. ROI pooling is used by Faster R-CNN to produce fixed-size feature maps from arbitrary object proposals (bounding boxes).

Code is located [here](tensorflow/core/user_ops).  