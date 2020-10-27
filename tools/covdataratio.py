import csv
import os
import os.path
import tarfile
from urllib.parse import urlparse

import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
from skimage import io
from PIL import Image

object_categories = ['normal','cov',  'other']
infofiles = []#[['traininfo0.csv', 'testinfo0.csv'],['traininfo1.csv', 'testinfo1.csv'],['traininfo2.csv', 'testinfo2.csv'],['traininfo3.csv', 'testinfo3.csv'],['traininfo4.csv', 'testinfo4.csv']]

for isplit in range(15):
    infofiles.append(['traininfo_ratio_' + str(isplit) +'.csv', 'testinfo_ratio_' + str(isplit) +'.csv'])
print(infofiles)

def read_image_label(file, combine,classnum):
    print('[dataset] read ' + file)
    df = pd.read_csv(file)
    if combine and classnum == 2:
        df.loc[df['label'] == 2, 'label'] = 1

    if classnum == 2:
        df = df[(df['label'] == 0)|(df['label'] == 1)]
    filenamelist = df['filename'].tolist()
    label = df['label'].tolist()
    #tempdata = {filenamelist[i]:label[i] for i in range(len(filenamelist))}
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
        #print(os.path.join(self.path_images, self.folder, path))
        #imgright = Image.open(os.path.join(self.path_images, 'right', path)).convert('RGB')
        #imgleft = Image.open(os.path.join(self.path_images, 'left', path)).convert('RGB')
        if self.transform is not None:
            imgall = self.transform(imgall)
            #imgright = self.transform(imgright)
            #imgleft = self.transform(imgleft)

        return {'target':target, 'imgall':imgall} #'imgright':imgright, 'imgleft':imgleft

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)

# class vocdataloader(data.Dataset):
#     def __init__(self, root, set, transform=None, combine=True):
#         self.root = root
#         self.path_images = os.path.join(root, 'images')
#         self.transform = transform
#         self.classes = object_categories
#         if set == 'train':
#             file_csv = root + infofiles[0]
#         else:
#             file_csv = root + infofiles[1]
#         self.images, self.labels = read_image_label(file_csv, combine)
#
#         print('[dataset] COVID19 classification set=%s number of classes=%d  number of images=%d' % (
#             set, len(self.classes), len(self.images)))
#
#     def __getitem__(self, index):
#         path, target = self.images[index], self.labels[index]
#         imgall = Image.open(os.path.join(self.path_images, 'all', path)).convert('RGB')
#         #imgright = Image.open(os.path.join(self.path_images, 'right', path)).convert('RGB')
#         #imgleft = Image.open(os.path.join(self.path_images, 'left', path)).convert('RGB')
#         if self.transform is not None:
#             imgall = self.transform(imgall)
#             #imgright = self.transform(imgright)
#             #imgleft = self.transform(imgleft)
#
#         return {'target':target, 'imgall':imgall} #'imgright':imgright, 'imgleft':imgleft
#
#     def __len__(self):
#         return len(self.images)
#
#     def get_number_classes(self):
#         return len(self.classes)


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

