import csv
import os
import os.path

import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

object_categories = ['normal','cov',  'other']
infofiles = []
for isplit in range(15):
    infofiles.append(['traininfo' + str(isplit) +'.csv', 'testinfo' + str(isplit) +'.csv'])


def read_image_label(file, combine,classnum):
    print('[dataset] read ' + file)
    df = pd.read_csv(file)
    if combine and classnum == 2:
        df.loc[df['label'] == 2, 'label'] = 1

    if classnum == 2:
        df = df[(df['label'] == 0)|(df['label'] == 1)]
    filenamelist = df['filename'].tolist()
    label = df['label'].tolist()
    return filenamelist, label

class vocdataloader(data.Dataset):
    def __init__(self, root, set, transform=None, combine=True, split=0, classnum=2,folder='all'):
        self.root = root
        self.path_images = os.path.join(root, 'images')
        self.transform = transform
        self.classes = object_categories
        self.folder = folder
        if set == 'train':
            file_csv = root + infofiles[split][0]
        else:
            file_csv = root + infofiles[split][1]
        self.images, self.labels = read_image_label(file_csv, combine,classnum)

        print('[dataset] COVID19 classification set=%s number of classes=%d  number of images=%d' % (
            set, len(self.classes), len(self.images)))

    def __getitem__(self, index):
        path, target = self.images[index], self.labels[index]
        imgall = Image.open(os.path.join(self.path_images, self.folder, path)).convert('RGB')
        if self.transform is not None:
            imgall = self.transform(imgall)

        return {'target':target, 'imgall':imgall} #'imgright':imgright, 'imgleft':imgleft

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)


class vocdataloader_trip(data.Dataset):
    def __init__(self, root, set, transform=None, combine=True, split=0, classnum=2):
        self.root = root
        self.path_images = os.path.join(root, 'images')
        self.transform = transform
        self.classes = object_categories
        if set == 'train':
            file_csv = root + infofiles[split][0]
        else:
            file_csv = root + infofiles[split][1]
        self.images, self.labels = read_image_label(file_csv, combine,classnum)

        print('[dataset] COVID19 classification set=%s number of classes=%d  number of images=%d' % (
            set, len(self.classes), len(self.images)))

    def __getitem__(self, index):
        path, target = self.images[index], self.labels[index]
        imgall = Image.open(os.path.join(self.path_images, 'all', path)).convert('RGB')
        imgright = Image.open(os.path.join(self.path_images, 'right', path)).convert('RGB')
        imgleft = Image.open(os.path.join(self.path_images, 'left', path)).convert('RGB')
        if self.transform is not None:
            imgall = self.transform(imgall)
            imgright = self.transform(imgright)
            imgleft = self.transform(imgleft)
        comimage = np.concatenate((imgall,imgright,imgleft),axis=0)
        return {'target':target, 'imgall':comimage}

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)



class vocdataloader_patch(data.Dataset):
    def __init__(self, root, set, transform=None, combine=True, split=0, classnum=2, randomcrops=100):
        self.root = root
        self.path_images = os.path.join(root, 'images')
        self.transform = transform
        self.classes = object_categories
        self.randomcrops = randomcrops
        self.set = set
        if set == 'train':
            file_csv = root + infofiles[split][0]
        else:
            file_csv = root + infofiles[split][1]
        self.images, self.labels = read_image_label(file_csv, combine,classnum)

        print('[dataset] COVID19 classification set=%s number of classes=%d  number of images=%d' % (
            set, len(self.classes), len(self.images)))

    def __getitem__(self, index):
        path, target = self.images[index], self.labels[index]
        imgall = Image.open(os.path.join(self.path_images, 'all', path)).convert('RGB')
        imgright = Image.open(os.path.join(self.path_images, 'right', path)).convert('RGB')
        imgleft = Image.open(os.path.join(self.path_images, 'left', path)).convert('RGB')
        comimage = []
        if (self.transform is not None) and (self.set == 'train'):
            imgall = self.transform[0](imgall)
            imgright = self.transform[1](imgright)
            imgleft = self.transform[1](imgleft)
            comimage = np.concatenate((imgall, imgright, imgleft), axis=0)
        elif (self.transform is not None) and (self.set == 'val'):
            comimage = torch.zeros([self.randomcrops,9,224,224])
            for icrop in range(self.randomcrops):
                comimage[icrop, 0:3, :, :] = self.transform[0](imgall)
                comimage[icrop, 3:6, :, :] = self.transform[1](imgright)
                comimage[icrop, 6:9, :, :] = self.transform[1](imgleft)

        return {'target': target, 'imgall': comimage}

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)