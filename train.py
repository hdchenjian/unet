import sys
sys.path.append("/opt/ego/caffe-rcnn-face-ssd/python")
import caffe
import numpy as np

# export LD_LIBRARY_PATH=/opt/ego/caffe-rcnn-face-ssd/lib:/opt/ego/boost_1_61/lib/:/opt/ego/opencv_3_3/lib

caffe.set_mode_gpu()
caffe.set_device(0)
solver = caffe.AdamSolver('solver.prototxt')

epochs = 10
train_iter_per_epoch = 3000

g_avg_loss = []
best_acc = 0
for i in range(epochs):
    avg_loss = np.zeros(train_iter_per_epoch)
    for j in range(train_iter_per_epoch):
        solver.step(1)
        avg_loss[j] = solver.net.blobs['loss'].data
        if j % 50 == 0:
            mean_loss = avg_loss.sum()/(j+1)
            g_avg_loss.append(mean_loss)
            print('epoch: %d, iters: %d, loss: %.4f, finished: %.2f' %
                  (i+1, i*train_iter_per_epoch+j, mean_loss, 100.0*(j+1)/train_iter_per_epoch))
