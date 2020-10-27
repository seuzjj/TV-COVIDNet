import torch.nn as nn
import torchvision.models as models
import torch

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class other_COVNet(nn.Module):
    def __init__(self, model, num_classes=2):
        super(other_COVNet, self).__init__()
        model.classifier[6] = nn.Linear(4096,num_classes)
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return x

class OtherNet_trip_max(nn.Module):
    def __init__(self, model1,model2, model3, num_classes=2):
        super(OtherNet_trip_max, self).__init__()
        num_ftrs = 4096
        model1.classifier[6] = Identity()
        model2.classifier[6] = Identity()
        model3.classifier[6] = Identity()
        self.features1 = model1
        self.features2 = model2
        self.features3 = model3
        self.classifier = nn.Linear(num_ftrs, num_classes)
    def forward(self, x):
        x1 = self.features1(x[:, 0:3, :, :])
        x2 = self.features2(x[:, 3:6, :, :])
        x3 = self.features3(x[:, 6:9, :, :])
        x = torch.max(x1, x2)
        x = torch.max(x, x3)
        x = self.classifier(x)
        return x

class OtherNet_trip_mean(nn.Module):
    def __init__(self, model1,model2, model3, num_classes=2):
        super(OtherNet_trip_mean, self).__init__()
        num_ftrs = 4096
        model1.classifier[6] = Identity()
        model2.classifier[6] = Identity()
        model3.classifier[6] = Identity()
        self.features1 = model1
        self.features2 = model2
        self.features3 = model3
        self.classifier = nn.Linear(num_ftrs, num_classes)
    def forward(self, x):
        x1 = self.features1(x[:, 0:3, :, :])
        x2 = self.features2(x[:, 3:6, :, :])
        x3 = self.features3(x[:, 6:9, :, :])
        x = torch.mean(torch.stack((x1,x2,x3),dim=2),dim=2)
        x = self.classifier(x)
        return x

class OtherNet_trip_cat(nn.Module):
    def __init__(self, model1,model2, model3, num_classes=2):
        super(OtherNet_trip_cat, self).__init__()
        num_ftrs = 4096
        model1.classifier[6] = Identity()
        model2.classifier[6] = Identity()
        model3.classifier[6] = Identity()
        self.features1 = model1
        self.features2 = model2
        self.features3 = model3
        self.classifier = nn.Linear(num_ftrs*3, num_classes)
    def forward(self, x):
        x1 = self.features1(x[:, 0:3, :, :])
        x2 = self.features2(x[:, 3:6, :, :])
        x3 = self.features3(x[:, 6:9, :, :])
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.classifier(x)
        return x

class OtherNet_trip_score(nn.Module):
    def __init__(self, model1,model2, model3, num_classes=2):
        super(OtherNet_trip_score, self).__init__()
        model1.classifier[6] = nn.Linear(4096, num_classes)
        model2.classifier[6] = nn.Linear(4096, num_classes)
        model3.classifier[6] = nn.Linear(4096, num_classes)
        self.features1 = model1
        self.features2 = model2
        self.features3 = model3
    def forward(self, x):
        x1 = self.features1(x[:, 0:3, :, :])
        x2 = self.features2(x[:, 3:6, :, :])
        x3 = self.features3(x[:, 6:9, :, :])
        x = torch.mean(torch.stack((x1, x2, x3), dim=2), dim=2)
        return x

class OtherNet_double_score(nn.Module):
    def __init__(self, model1,model2, num_classes=2):
        super(OtherNet_double_score, self).__init__()
        model1.classifier[6] = nn.Linear(4096, num_classes)
        model2.classifier[6] = nn.Linear(4096, num_classes)
        self.features1 = model1
        self.features2 = model2

    def forward(self, x):
        x1 = self.features1(x[:, 3:6, :, :])
        x2 = self.features2(x[:, 6:9, :, :])
        x = torch.mean(torch.stack((x1, x2), dim=2), dim=2)
        return x

class OtherNet_trip_adaptive(nn.Module):
    def __init__(self, model1,model2, model3, num_classes=2):
        super(OtherNet_trip_adaptive, self).__init__()
        model1.classifier[6] = nn.Linear(4096, num_classes)
        model2.classifier[6] = nn.Linear(4096, num_classes)
        model3.classifier[6] = nn.Linear(4096, num_classes)
        self.features1 = model1
        self.features2 = model2
        self.features3 = model3
        self.weight = nn.Parameter(torch.tensor([1 / 3.0, 1 / 3.0, 1 / 3.0], requires_grad=True))
        
    def forward(self, x):
        x1 = self.features1(x[:, 0:3, :, :])
        x2 = self.features2(x[:, 3:6, :, :])
        x3 = self.features3(x[:, 6:9, :, :])
        self.weight.data = torch.max(self.weight.data,torch.zeros_like(self.weight.data))
        self.weight.data = self.weight.data / torch.sum(self.weight.data)
        x  = torch.matmul(torch.stack((x1,x2,x3),dim=2),self.weight)
        return x
    
class OtherNet_trip_scoremax(nn.Module):
    def __init__(self, model1,model2, model3, num_classes=2):
        super(OtherNet_trip_scoremax, self).__init__()
        model1.classifier[6] = nn.Linear(4096, num_classes)
        model2.classifier[6] = nn.Linear(4096, num_classes)
        model3.classifier[6] = nn.Linear(4096, num_classes)
        self.features1 = model1
        self.features2 = model2
        self.features3 = model3
    def forward(self, x):
        x1 = self.features1(x[:, 0:3, :, :])
        x2 = self.features2(x[:, 3:6, :, :])
        x3 = self.features3(x[:, 6:9, :, :])
        x = torch.max(x1, x2)
        x = torch.max(x, x3)
        return x

def alex_trip_max(num_classes=2, pretrained=True):
    model1 = models.alexnet(pretrained)
    model2 = models.alexnet(pretrained)
    model3 = models.alexnet(pretrained)
    return OtherNet_trip_max(model1,model2, model3, num_classes)

def alex_trip_mean(num_classes=2, pretrained=True):
    model1 = models.alexnet(pretrained)
    model2 = models.alexnet(pretrained)
    model3 = models.alexnet(pretrained)
    return OtherNet_trip_mean(model1,model2, model3, num_classes)

def alex_trip_cat(num_classes=2, pretrained=True):
    model1 = models.alexnet(pretrained)
    model2 = models.alexnet(pretrained)
    model3 = models.alexnet(pretrained)
    return OtherNet_trip_cat(model1,model2, model3, num_classes)

def alex_trip_score(num_classes=2, pretrained=True):
    model1 = models.alexnet(pretrained)
    model2 = models.alexnet(pretrained)
    model3 = models.alexnet(pretrained)
    return OtherNet_trip_score(model1,model2, model3, num_classes)

def alex_double_score(num_classes=2, pretrained=True):
    model1 = models.alexnet(pretrained)
    model2 = models.alexnet(pretrained)
    return OtherNet_double_score(model1,model2, num_classes)

def alex_trip_scoremax(num_classes=2, pretrained=True):
    model1 = models.alexnet(pretrained)
    model2 = models.alexnet(pretrained)
    model3 = models.alexnet(pretrained)
    return OtherNet_trip_scoremax(model1,model2, model3, num_classes)

def alex_trip_adaptive(num_classes=2, pretrained=True):
    model1 = models.alexnet(pretrained)
    model2 = models.alexnet(pretrained)
    model3 = models.alexnet(pretrained)
    return OtherNet_trip_adaptive(model1,model2, model3, num_classes)

####################
def vgg16_trip_max(num_classes=2, pretrained=True):
    model1 = models.vgg16(pretrained)
    model2 = models.vgg16(pretrained)
    model3 = models.vgg16(pretrained)
    return OtherNet_trip_max(model1,model2, model3, num_classes)

def vgg16_trip_mean(num_classes=2, pretrained=True):
    model1 = models.vgg16(pretrained)
    model2 = models.vgg16(pretrained)
    model3 = models.vgg16(pretrained)
    return OtherNet_trip_mean(model1,model2, model3, num_classes)

def vgg16_trip_cat(num_classes=2, pretrained=True):
    model1 = models.vgg16(pretrained)
    model2 = models.vgg16(pretrained)
    model3 = models.vgg16(pretrained)
    return OtherNet_trip_cat(model1,model2, model3, num_classes)

def vgg16_trip_score(num_classes=2, pretrained=True):
    model1 = models.vgg16(pretrained)
    model2 = models.vgg16(pretrained)
    model3 = models.vgg16(pretrained)
    return OtherNet_trip_score(model1,model2, model3, num_classes)

def vgg16_double_score(num_classes=2, pretrained=True):
    model1 = models.vgg16(pretrained)
    model2 = models.vgg16(pretrained)
    return OtherNet_double_score(model1,model2, num_classes)

def vgg16_trip_scoremax(num_classes=2, pretrained=True):
    model1 = models.vgg16(pretrained)
    model2 = models.vgg16(pretrained)
    model3 = models.vgg16(pretrained)
    return OtherNet_trip_scoremax(model1,model2, model3, num_classes)


def vgg16_trip_adaptive(num_classes=2, pretrained=True):
    model1 = models.vgg16(pretrained)
    model2 = models.vgg16(pretrained)
    model3 = models.vgg16(pretrained)
    return OtherNet_trip_adaptive(model1,model2, model3, num_classes)

####################
def vgg19_trip_max(num_classes=2, pretrained=True):
    model1 = models.vgg19(pretrained)
    model2 = models.vgg19(pretrained)
    model3 = models.vgg19(pretrained)
    return OtherNet_trip_max(model1,model2, model3, num_classes)

def vgg19_trip_mean(num_classes=2, pretrained=True):
    model1 = models.vgg19(pretrained)
    model2 = models.vgg19(pretrained)
    model3 = models.vgg19(pretrained)
    return OtherNet_trip_mean(model1,model2, model3, num_classes)

def vgg19_trip_cat(num_classes=2, pretrained=True):
    model1 = models.vgg19(pretrained)
    model2 = models.vgg19(pretrained)
    model3 = models.vgg19(pretrained)
    return OtherNet_trip_cat(model1,model2, model3, num_classes)

def vgg19_trip_score(num_classes=2, pretrained=True):
    model1 = models.vgg19(pretrained)
    model2 = models.vgg19(pretrained)
    model3 = models.vgg19(pretrained)
    return OtherNet_trip_score(model1,model2, model3, num_classes)

def vgg19_double_score(num_classes=2, pretrained=True):
    model1 = models.vgg19(pretrained)
    model2 = models.vgg19(pretrained)
    return OtherNet_double_score(model1,model2, num_classes)

def vgg19_trip_scoremax(num_classes=2, pretrained=True):
    model1 = models.vgg19(pretrained)
    model2 = models.vgg19(pretrained)
    model3 = models.vgg19(pretrained)
    return OtherNet_trip_scoremax(model1,model2, model3, num_classes)

def vgg19_trip_adaptive(num_classes=2, pretrained=True):
    model1 = models.vgg19(pretrained)
    model2 = models.vgg19(pretrained)
    model3 = models.vgg19(pretrained)
    return OtherNet_trip_adaptive(model1,model2, model3, num_classes)

def alex_COVNet(num_classes=2, pretrained=True):
    model = models.alexnet(pretrained)
    return other_COVNet(model, num_classes)

def vgg16_COVNet(num_classes=2, pretrained=True):
    model = models.vgg16(pretrained)

    return other_COVNet(model, num_classes)

def vgg19_COVNet(num_classes=2, pretrained=True):
    model = models.vgg19(pretrained)

    return other_COVNet(model, num_classes)

def googlenet_COVNet(num_classes=2, pretrained=True):
    model = models.googlenet(pretrained)

    return COVNet(model, num_classes)

class GoogleNet_trip_max(nn.Module):
    def __init__(self, model1,model2, model3, num_classes=2):
        super(GoogleNet_trip_max, self).__init__()

        num_ftrs = model1.fc.in_features
        model1.fc = Identity()
        model2.fc = Identity()
        model3.fc = Identity()
        self.features1 = model1
        self.features2 = model2
        self.features3 = model3

        self.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x1 = self.features1(x[:, 0:3, :, :])
        x2 = self.features2(x[:, 3:6, :, :])
        x3 = self.features3(x[:, 6:9, :, :])
        x = torch.max(x1, x2)
        x = torch.max(x, x3)
        x = self.classifier(x)
        return x

class GoogleNet_trip_mean(nn.Module):
    def __init__(self, model1,model2, model3, num_classes=2):
        super(GoogleNet_trip_mean, self).__init__()

        num_ftrs = model1.fc.in_features
        model1.fc = Identity()
        model2.fc = Identity()
        model3.fc = Identity()
        self.features1 = model1
        self.features2 = model2
        self.features3 = model3

        self.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x1 = self.features1(x[:, 0:3, :, :])
        x2 = self.features2(x[:, 3:6, :, :])
        x3 = self.features3(x[:, 6:9, :, :])
        x = torch.mean(torch.stack((x1,x2,x3),dim=2),dim=2)
        x = self.classifier(x)
        return x

class GoogleNet_trip_cat(nn.Module):
    def __init__(self, model1,model2, model3, num_classes=2):
        super(GoogleNet_trip_cat, self).__init__()

        num_ftrs = model1.fc.in_features
        model1.fc = Identity()
        model2.fc = Identity()
        model3.fc = Identity()
        self.features1 = model1
        self.features2 = model2
        self.features3 = model3

        self.classifier = nn.Linear(num_ftrs*3, num_classes)

    def forward(self, x):
        x1 = self.features1(x[:, 0:3, :, :])
        x2 = self.features2(x[:, 3:6, :, :])
        x3 = self.features3(x[:, 6:9, :, :])
        x = torch.cat((x1,x2,x3),dim=1)
        x = self.classifier(x)
        return x

class GoogleNet_trip_score(nn.Module):
    def __init__(self, model1,model2, model3, num_classes=2):
        super(GoogleNet_trip_score, self).__init__()

        num_ftrs = model1.fc.in_features
        model1.fc = nn.Linear(num_ftrs, num_classes)
        model2.fc = nn.Linear(num_ftrs, num_classes)
        model3.fc = nn.Linear(num_ftrs, num_classes)
        self.features1 = model1
        self.features2 = model2
        self.features3 = model3

    def forward(self, x):
        x1 = self.features1(x[:, 0:3, :, :])
        x2 = self.features2(x[:, 3:6, :, :])
        x3 = self.features3(x[:, 6:9, :, :])
        x = torch.mean(torch.stack((x1,x2,x3),dim=2),dim=2)
        return x

class GoogleNet_double_score(nn.Module):
    def __init__(self, model1,model2, num_classes=2):
        super(GoogleNet_double_score, self).__init__()
        num_ftrs = model1.fc.in_features
        model1.fc = nn.Linear(num_ftrs, num_classes)
        model2.fc = nn.Linear(num_ftrs, num_classes)
        self.features1 = model1
        self.features2 = model2

    def forward(self, x):
        x1 = self.features1(x[:, 3:6, :, :])
        x2 = self.features2(x[:, 6:9, :, :])
        x = torch.mean(torch.stack((x1,x2),dim=2),dim=2)
        return x

class GoogleNet_trip_scoremax(nn.Module):
    def __init__(self, model1,model2, model3, num_classes=2):
        super(GoogleNet_trip_scoremax, self).__init__()

        num_ftrs = model1.fc.in_features
        model1.fc = nn.Linear(num_ftrs, num_classes)
        model2.fc = nn.Linear(num_ftrs, num_classes)
        model3.fc = nn.Linear(num_ftrs, num_classes)
        self.features1 = model1
        self.features2 = model2
        self.features3 = model3

    def forward(self, x):
        x1 = self.features1(x[:, 0:3, :, :])
        x2 = self.features2(x[:, 3:6, :, :])
        x3 = self.features3(x[:, 6:9, :, :])
        x = torch.max(x1, x2)
        x = torch.max(x, x3)
        return x

class GoogleNet_trip_adaptive(nn.Module):
    def __init__(self, model1,model2, model3, num_classes=2):
        super(GoogleNet_trip_adaptive, self).__init__()

        num_ftrs = model1.fc.in_features
        model1.fc = nn.Linear(num_ftrs, num_classes)
        model2.fc = nn.Linear(num_ftrs, num_classes)
        model3.fc = nn.Linear(num_ftrs, num_classes)
        self.features1 = model1
        self.features2 = model2
        self.features3 = model3
        self.weight = nn.Parameter(torch.tensor([1 / 3.0, 1 / 3.0, 1 / 3.0], requires_grad=True))

    def forward(self, x):
        x1 = self.features1(x[:, 0:3, :, :])
        x2 = self.features2(x[:, 3:6, :, :])
        x3 = self.features3(x[:, 6:9, :, :])
        self.weight.data = torch.max(self.weight.data,torch.zeros_like(self.weight.data))
        self.weight.data = self.weight.data / torch.sum(self.weight.data)
        x  = torch.matmul(torch.stack((x1,x2,x3),dim=2),self.weight)
        return x

def Googlemodel_trip_score(num_classes=2, pretrained=True):
    model1 = models.googlenet(pretrained)
    model2 = models.googlenet(pretrained)
    model3 = models.googlenet(pretrained)
    return GoogleNet_trip_score(model1,model2, model3, num_classes)


def Googlemodel_double_score(num_classes=2, pretrained=True):
    model1 = models.googlenet(pretrained)
    model2 = models.googlenet(pretrained)
    return GoogleNet_double_score(model1,model2, num_classes)

def Googlemodel_trip_mean(num_classes=2, pretrained=True):
    model1 = models.googlenet(pretrained)
    model2 = models.googlenet(pretrained)
    model3 = models.googlenet(pretrained)
    return GoogleNet_trip_mean(model1,model2, model3, num_classes)

def Googlemodel_trip_max(num_classes=2, pretrained=True):
    model1 = models.googlenet(pretrained)
    model2 = models.googlenet(pretrained)
    model3 = models.googlenet(pretrained)
    return GoogleNet_trip_max(model1,model2, model3, num_classes)

def Googlemodel_trip_cat(num_classes=2, pretrained=True):
    model1 = models.googlenet(pretrained)
    model2 = models.googlenet(pretrained)
    model3 = models.googlenet(pretrained)
    return GoogleNet_trip_cat(model1,model2, model3, num_classes)
def Googlemodel_trip_scoremax(num_classes=2, pretrained=True):
    model1 = models.googlenet(pretrained)
    model2 = models.googlenet(pretrained)
    model3 = models.googlenet(pretrained)
    return GoogleNet_trip_scoremax(model1,model2, model3, num_classes)
def Googlemodel_trip_adaptive(num_classes=2, pretrained=True):
    model1 = models.googlenet(pretrained)
    model2 = models.googlenet(pretrained)
    model3 = models.googlenet(pretrained)
    return GoogleNet_trip_adaptive(model1,model2, model3, num_classes)
