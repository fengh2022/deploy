import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from utils.params import arg
import os
from glob import glob
import cv2
from PIL import Image



CLS2IDX = {'Cat':0, 'Dog':1}
CLASS_NAMES = ['Cat', 'Dog']

class CatDogDataset(Dataset):
    def __init__(self, rootdir, is_training=False):
        self.rootdir = rootdir
        self.is_training = is_training

        sub_folder = 'train' if self.is_training else 'test'
        self.datadir = os.path.join(self.rootdir, sub_folder)
        self.images = []
        self.labels = []

        for idx, class_name in enumerate(CLASS_NAMES):
            for img_path in glob(os.path.join(self.datadir, class_name, '*.jpg')):
                self.images.append(img_path)
                self.labels.append(idx)

        print(sub_folder, len(self.labels))

    def __len__(self):
        return len(self.labels)

    def process(self, img):
        img = cv2.resize(img, (arg.img_w, arg.img_h))
        img = img/255.0
        img = np.transpose(img, (2,0,1))

        return img

    def imread(self, img_path):
        img = Image.open(img_path).convert('RGB')  # to bgr
        img = np.array(img)

        return img

    def __getitem__(self, item):
        while True:
            try:
                img_ori = self.imread(self.images[item])
                img = self.process(img_ori)
                lab = self.labels[item]
                break
            except Exception as e:
                item += 1
                # print(e)

        return img, lab



if __name__=='__main__':
    dataset = CatDogDataset(rootdir=arg.root_dir, is_training=True)
    dataloader = DataLoader(
        dataset,
        batch_size=arg.batch_size,
        num_workers=arg.num_workers,
        shuffle=True
    )

    for batch_img, batch_lab in dataloader:
        print(batch_img.min(), batch_lab.min())


