import sys
sys.path.append("/opt/ego/caffe-rcnn-face-ssd/python")
import caffe
import numpy as np
import cv2

caffe.set_mode_gpu()
caffe.set_device(0)

def load_image(image_path):
    im = cv2.imread(image_path)
    #print('im', im.shape)
    im = cv2.resize(im,(572,572))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = np.array(im, np.float64)
    im /= 255.0
    im -= 0.5
    return im[np.newaxis, :]

def load_mask(image_path):
    outimg = np.empty((2,572,572))
    im = cv2.imread(image_path)
    #print('label ', im.shape)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.resize(im,(572,572))
    ret, img = cv2.threshold(im, 0.5, 1.0, cv2.THRESH_BINARY)
    return img[np.newaxis, :]

img = load_image('data/test/0.png')
print('img', img.shape)
label = load_mask('data/test/0_predict.png')
print('label', label.shape)

net = caffe.Net("train_val.prototxt", "weight/solver_iter_30000.caffemodel", caffe.TEST)
net.blobs['img'].data[...] = img
net.blobs['mask'].data[...] = label

net.forward()
print(type(net.blobs['score'].data), net.blobs['score'].data.shape)
output = net.blobs['score'].data[0][0] > 0.5
output = output*255
cv2.imwrite('output.jpg', output)


print(net.blobs['loss'].data)
