# Power-Line-Assets-Detection-with-YOLOv3 (in progress, not yet completed)

This project aims to detect power line assets such as transmission tower,
insulator, spacer, tower plate, and Stockbridge damper from aerial images using the YOLOv3 object detection algorithm on STN-PLAD DATASET. 
The goal is to assist power companies in identifying and monitoring their assets, which can improve maintenance and reduce downtime.

Get Dataset: [STN PLAD: A Dataset for Multi-Size Power Line Assets Detection in High-Resolution UAV Images](https://github.com/andreluizbvs/PLAD).

## Completed tasks:
1. Implemented YOLOv3 architecture (Darknet-53 as Backbone) for both training and prediction purposes.
2. Implemented YOLO-Loss.
3. Implemented Intersection Over Union.
4. Implemented Non-max suppression.
5. Converted Bounding Boxes to yolo Bounding Boxes across 3 scales and each with grid cell with 3 anchors.

## TO DO:
1. write a generator class, as the dataset size is so big with each image size of 3468x5472, can't fit the whole data onto memory.
2. find bugs (if any).
3. implement k-means to find best anchors (if necessary).

## References:
1. Andrew Ng's Convolutional Neural Networks Course on Coursera.
2. [Redmon et. al. in You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
3. [Redmon et al. in YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242)
4. [Redmon et al. in YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)
5. [github.com/zzh8829/yolov3-tf2](https://github.com/zzh8829/yolov3-tf2)
6. [github.com/qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3)

