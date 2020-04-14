import sys
sys.path.append("/opt/ego/caffe-rcnn-face-ssd/python")
import caffe
import numpy as np
import cv2

caffe.set_mode_gpu()
caffe.set_device(0)

def load_image(imname):
    im = cv2.imread(imname)
    im = cv2.resize(im,(640, 480))
    cv2.imwrite('output_.jpg', im)
    print('load_image', im.shape)
    im = np.array(im, np.float32)
    im /= 255.0
    im -= 0.5
    im = im.transpose((2,0,1))
    print('load_image', im.shape)
    return im

net = caffe.Net("train_my.prototxt", "weight/solver_iter_4000.caffemodel", caffe.TEST)

img = load_image('../tf/tf_unet/data/muyuan/test/VISIBLE_PIC_41.jpg')
#img = load_image('../tf/tf_unet/data/muyuan/train/2.png')
net.blobs['data'].data[...] = img

net.forward()
print(type(net.blobs['predict'].data), net.blobs['predict'].data.shape)
output = net.blobs['predict'].data[0][1]
print(output.shape)
output = output*255
cv2.imwrite('output.jpg', output)
