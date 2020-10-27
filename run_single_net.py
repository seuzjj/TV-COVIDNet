from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import time
import copy
from tools.model import resnet50_COVNet, resnet101_COVNet, resnet152_COVNet

from tools.covdata import vocdataloader
from sklearn.metrics import confusion_matrix

import sys
import random
import pickle


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

            print(confusion_matrix(allgt, allpre))

            if phase == 'val':
                epoch_loss_test.append(epoch_loss)
            else:
                epoch_loss_train.append(epoch_loss)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                bestconfusion = confusion_matrix(allgt, allpre)
                best_model_wts = copy.deepcopy(model.state_dict())
                best_pre = alloutputs

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    overall_best.append(best_acc.item())
    best_confution_matrix.append(bestconfusion)
    print(best_confution_matrix)
    all_best_pre.append(allgt)
    all_best_pre.append(best_pre)
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


data_dir = './datasets/COVID19/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


resnets = ['res50', 'res101', 'res152']
for fold in ['all']: 
    print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    print(fold)

    for numclasses in [2, 3]:

        for ires in [0, 1, 2]:

            logfilename = './results/Singlenet_numclasses_' + str(numclasses) \
                          + resnets[ires] + fold + '_' + str(random.random()) + '.txt'
            orig_stdout = sys.stdout
            file = open(logfilename + '.pl', 'wb')
            logfile = open(logfilename, 'w')
            sys.stdout = logfile

            best_confution_matrix = []
            all_best_pre = []
            overall_best = []

            for isplit in range(15):
                print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
                print(isplit)

                epoch_loss_train = []
                epoch_loss_test = []

                if ires == 0:
                    model_ft = resnet50_COVNet(num_classes=numclasses)
                elif ires == 1:
                    model_ft = resnet101_COVNet(num_classes=numclasses)
                elif ires == 2:
                    model_ft = resnet152_COVNet(num_classes=numclasses)

                model_ft = model_ft.to(device)

                criterion = nn.CrossEntropyLoss()

                # Observe that all parameters are being optimized
                optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)

                # Decay LR by a factor of 0.1 every 7 epochs
                exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)

                image_datasets = {
                    x: vocdataloader(data_dir, x, transform=data_transforms[x], combine=False, split=isplit,
                                     classnum=numclasses, folder=fold)
                    for x in ['train', 'val']}

                dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                              shuffle=True, num_workers=4)
                               for x in ['train', 'val']}
                dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
                class_names = image_datasets['train'].classes

                model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                       num_epochs=100)

            logfile.close()
            pickle.dump(all_best_pre, file)
            pickle.dump(best_confution_matrix, file)

            file.close()

            sys.stdout = orig_stdout


