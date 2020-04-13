import caffe
import numpy as np
import cv2
import numpy.random as random

class DataLayer(caffe.Layer):

    def setup(self, bottom, top):

        self.imgdir = "../tf/tf_unet/data/muyuan/train/"
        self.maskdir = "../tf/tf_unet/data/muyuan/train/"
        self.imgtxt = "../tf/tf_unet/data/muyuan/train.txt"
        self.random = True
        self.seed = None

        if len(top) != 2:
            raise Exception("Need to define two tops: data and mask.")

        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        self.lines = []
        for _line in open(self.imgtxt, 'r').readlines():
            _line = _line.strip('\n')
            self.lines.append(_line)
        self.idx = 0

        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.lines) - 1)

    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = self.load_image(self.idx)
        self.mask = self.load_mask(self.idx)
        # reshape tops to fit (leading 1 is for batch dimension)
        #print('input shape', self.data.shape, self.mask.shape)
        #top[0].reshape(1, 3, 480, 640)
        #top[1].reshape(1, 1, 480, 640)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.mask.shape)

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.mask

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.lines) - 1)
        else:
            self.idx += 1
            if self.idx == len(self.lines):
                self.idx = 0

    def backward(self, top, propagate_down, bottom):
        pass

    def load_image(self, idx):
        imname = self.imgdir + self.lines[idx]
        #print('load img', imname)
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

    def load_mask(self, idx):
        imname = self.maskdir + self.lines[idx].replace('.png', '_mask.png')
        #print('load mask', imname)
        im = cv2.imread(imname, cv2.IMREAD_GRAYSCALE)
	im = cv2.resize(im,(640, 480))
        im = np.array(im, np.bool)
        labels = np.zeros((2, 480, 640), dtype=np.float32)
        #print('sum', sum(im))
        labels[1, ...] = im
        labels[0, ...] = ~im
        ret = im[np.newaxis, :]
        return ret
