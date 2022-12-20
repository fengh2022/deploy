import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import os
import numpy as np
from PIL import Image

import torchvision.transforms as transforms

from utils.params import arg
from glob import glob


class CatDogCalibrator(trt.IInt8LegacyCalibrator):

    def __init__(self):
        trt.IInt8LegacyCalibrator.__init__(self)

        self.cache_file = os.path.join(arg.calibration_dir, 'calibration.cache')
        self.data_dir = arg.calibration_dir

        self.batch_size = arg.batch_size
        self.Channel = 3
        self.Height = arg.img_h
        self.Width = arg.img_w
        self.transform = transforms.Compose([
            transforms.Resize([self.Height, self.Width]),  # [h,w]
            transforms.ToTensor(),
        ])

        self.imgs = glob(os.path.join(self.data_dir, '*.jpg'))
        self.batch_idx = 0
        self.max_batch_idx = len(self.imgs) // self.batch_size
        self.data_size = trt.volume([self.batch_size, self.Channel, self.Height, self.Width]) * trt.float32.itemsize
        self.device_input = cuda.mem_alloc(self.data_size)

        self.imgs_num = len(self.imgs)

        print(f'calibration img num is {self.imgs_num}')

    def free(self):
        self.device_input.free()

    def next_batch(self):
        if self.batch_idx < self.max_batch_idx:
            batch_files = self.imgs[self.batch_idx * self.batch_size: (self.batch_idx + 1) * self.batch_size]
            batch_imgs = np.zeros((self.batch_size, self.Channel, self.Height, self.Width),
                                  dtype=np.float32)

            for i, f in enumerate(batch_files):

                img = Image.open(f)
                img = self.transform(img).numpy()

                # 有部分图片是灰度图
                if img.shape[0]==1:
                    img = np.repeat(img, 3, axis=0)
                assert (img.nbytes == self.data_size / self.batch_size), 'not valid img!' + f
                batch_imgs[i] = img

            self.batch_idx += 1
            print("batch:[{}/{}]".format(self.batch_idx, self.max_batch_idx))

            return np.ascontiguousarray(batch_imgs)
        else:
            return np.array([])

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        try:
            batch_imgs = self.next_batch()

            print(batch_imgs.shape)
            if batch_imgs.size == 0 or batch_imgs.size != self.batch_size * self.Channel * self.Height * self.Width:
                return None
            cuda.memcpy_htod(self.device_input, batch_imgs.astype(np.float32).ravel())

            return [self.device_input]
        except:
            return None

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

    def get_quantile(self):
        return 0.9999

    def get_regression_cutoff(self):
        return 1.0

    def read_histogram_cache(self, length):
        return None

    def write_histogram_cache(self, ptr, length):
        return None