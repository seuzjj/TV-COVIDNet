from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tools.model import resnet50_COVNet, resnet50_COVNet_trip_mean, resnet50_COVNet_trip_cat, \
    resnet50_COVNet_trip_score, resnet50_COVNet_trip_max, resnet50_trip_score_max, resnet50_COVNet_double_score

from tools.model import resnet101_COVNet, resnet101_COVNet_trip_mean, resnet101_COVNet_trip_cat, \
    resnet101_COVNet_trip_score, resnet101_COVNet_trip_max, resnet101_trip_score_max, resnet101_COVNet_double_score

from tools.model import resnet152_COVNet, resnet152_COVNet_trip_mean, resnet152_COVNet_trip_cat, \
    resnet152_COVNet_trip_score, resnet152_COVNet_trip_max, resnet152_trip_score_max, resnet152_COVNet_double_score

from tools.covdata import vocdataloader
from tools.covdata import vocdataloader_trip
from sklearn.metrics import confusion_matrix

import sys
import random
import pickle

plt.ion()  # interactive mode

gpu_device = "cuda:1"

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    bestconfusion = []
    best_pre = []
    best_label = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            allpre = []
            allgt = []
            alloutputs = []
            # Iterate over data.
            for isample in dataloaders[phase]:

                inputs = isample['imgall'].to(device)
                labels = isample['target'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    else:
                        alloutputs.extend(outputs.data.tolist())

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                allpre.extend(preds.tolist())
                allgt.extend(labels.data.tolist())
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'val':
                epoch_loss_test.append(epoch_loss)
            else:
                epoch_loss_train.append(epoch_loss)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                bestconfusion = confusion_matrix(allgt, allpre)
                best_pre = alloutputs
                best_label = allgt

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    best_accu_all.append(best_acc.item())
    print(best_accu_all)
    best_confution_matrix.append(bestconfusion)
    print(best_confution_matrix)
    all_best_pre.append(best_label)
    all_best_pre.append(best_pre)
    # load best model weights
    print(all_best_pre)
    model.load_state_dict(best_model_wts)
    return model


data_dir = './datasets/COVID19/'

device = torch.device(gpu_device if torch.cuda.is_available() else "cpu")


numclasses = 2
model_id = ['score', 'scoremax', 'mean', 'cat', 'max']
resnets = ['res50', 'res101', 'res152']

for ires in [1]:
    for imodel in [0]:

        logfile = './results/' + 'Trip_model_' + model_id[imodel] + \
                  '_net_' + resnets[ires] + '_numclasses_' + str(numclasses) + '_' + str(random.random()) + '.txt'
        logfilename = 'Trip_model_' + model_id[imodel] + \
                      '_net_' + resnets[ires] + '_numclasses_' + str(numclasses) + '_' + str(random.random()) + '.txt'
        file = open(logfile + '.pl', 'wb')
        orig_stdout = sys.stdout
        logfile = open(logfile, 'w')
        sys.stdout = logfile

        best_accu_all = []
        best_confution_matrix = []
        all_best_pre = []
        for isplit in range(15):

            epoch_loss_train = []
            epoch_loss_test = []

            model_ft = []

            if ires == 0:
                if imodel == 2:
                    model_ft = resnet50_COVNet_trip_mean(num_classes=numclasses)
                elif imodel == 3:
                    model_ft = resnet50_COVNet_trip_cat(num_classes=numclasses)
                elif imodel == 0:
                    model_ft = resnet50_COVNet_trip_score(num_classes=numclasses)
                elif imodel == 4:
                    model_ft = resnet50_COVNet_trip_max(num_classes=numclasses)
                elif imodel == 1:
                    model_ft = resnet50_trip_score_max(num_classes=numclasses)
                else:
                    print('unknown model')
                    continue
            elif ires == 1:
                if imodel == 2:
                    model_ft = resnet101_COVNet_trip_mean(num_classes=numclasses)
                elif imodel == 3:
                    model_ft = resnet101_COVNet_trip_cat(num_classes=numclasses)
                elif imodel == 0:
                    model_ft = resnet101_COVNet_trip_score(num_classes=numclasses)
                elif imodel == 4:
                    model_ft = resnet101_COVNet_trip_max(num_classes=numclasses)
                elif imodel == 1:
                    model_ft = resnet101_trip_score_max(num_classes=numclasses)
                else:
                    print('unknown model')
                    continue
            elif ires == 2:
                if imodel == 2:
                    model_ft = resnet152_COVNet_trip_mean(num_classes=numclasses)
                elif imodel == 3:
                    model_ft = resnet152_COVNet_trip_cat(num_classes=numclasses)
                elif imodel == 0:
                    model_ft = resnet152_COVNet_trip_score(num_classes=numclasses)
                elif imodel == 4:
                    model_ft = resnet152_COVNet_trip_max(num_classes=numclasses)
                elif imodel == 1:
                    model_ft = resnet152_trip_score_max(num_classes=numclasses)
                else:
                    print('unknown model')
                    continue

            model_ft = model_ft.to(device)

            criterion = nn.CrossEntropyLoss()

            # Observe that all parameters are being optimized
            optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)

            # Decay LR by a factor of 0.1 every 7 epochs
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)

            print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
            print(isplit)
            # global dataloaders, dataset_sizes, class_names

            image_datasets = {
                x: vocdataloader_trip(data_dir, x, transform=data_transforms[x], combine=False, split=isplit,
                                      classnum=numclasses) for x in ['train', 'val']}

            dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                          shuffle=True, num_workers=4)
                           for x in ['train', 'val']}
            dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
            class_names = image_datasets['train'].classes

            model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                   num_epochs=100)
            torch.save(model_ft.state_dict(), './savedmodel/' + logfilename + '_split_' + str(isplit))


        # dump information to that file
        pickle.dump(all_best_pre, file)
        pickle.dump(best_confution_matrix, file)

        file.close()
        sys.stdout = orig_stdout
        logfile.close()



