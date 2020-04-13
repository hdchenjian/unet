#!/bin/sh

export LD_LIBRARY_PATH=/opt/ego/caffe-rcnn-face-ssd/lib/:/opt/ego/boost_1_61/lib/:/opt/ego/opencv_3_3/lib/
/opt/ego/caffe-rcnn-face-ssd/bin/caffe train --solver=solver.prototxt
