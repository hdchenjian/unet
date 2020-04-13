import sys
sys.path.append("/opt/ego/caffe-rcnn-face-ssd/python")
import caffe
import numpy as np
import cv2

caffe.set_mode_gpu()
caffe.set_device(0)

def load_image(imname):
    im = cv2.imread(imname)
    #print('im', im.shape)
    #im = cv2.imread(imname)
    im = cv2.resize(im,(640, 480))
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = np.rollaxis(im, 2, 0)
    im = np.array(im, np.float32)
    im /= 255.0
    im -= 0.5
    #ret = im[np.newaxis, :]
    #print('load_image', im.shape)
    return im

def load_mask(imname):
    outimg = np.empty((2, 640, 480))
    im = cv2.imread(imname)
    #print('im', im.shape)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.resize(im,(640, 480))
    ret, im = cv2.threshold(im, 0.5, 1.0, cv2.THRESH_BINARY)
    #print('load_mask', im.shape)
    return im[np.newaxis, :]

net = caffe.Net("train_my.prototxt", "weight/solver_iter_1000.caffemodel", caffe.TEST)

img = load_image('../tf/tf_unet/data/muyuan/test/VISIBLE_PIC_41.jpg')
print('img', img.shape)
label = load_mask('data/test/0_predict.png')
#print('label', label.shape, label[0][0], sum(label[0]))

net.blobs['data'].data[...] = img
net.blobs['label'].data[...] = label

net.forward()
print(type(net.blobs['predict'].data), net.blobs['predict'].data.shape)
output = net.blobs['predict'].data[0][0]
#print(output.shape, sum(output))
output = output*255
cv2.imwrite('output.jpg', output)

#print(net.blobs['loss'].data)
