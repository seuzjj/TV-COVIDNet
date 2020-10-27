
import torch.nn as nn
import torchvision.models as models
import torch



class COVNet(nn.Module):

    def __init__(self, model, num_classes=2):
        super(COVNet, self).__init__()
        num_ftrs = model.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        model.fc = nn.Linear(num_ftrs, num_classes)
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return x


class COVNet_trip_max(nn.Module):

    def __init__(self, model1,model2, model3, num_classes=2):
        super(COVNet_trip_max, self).__init__()

        self.features1 = nn.Sequential(
            model1.conv1,
            model1.bn1,
            model1.relu,
            model1.maxpool,
            model1.layer1,
            model1.layer2,
            model1.layer3,
            model1.layer4,
            model1.avgpool)


        self.features2 = nn.Sequential(
            model2.conv1,
            model2.bn1,
            model2.relu,
            model2.maxpool,
            model2.layer1,
            model2.layer2,
            model2.layer3,
            model2.layer4,
            model2.avgpool)


        self.features3 = nn.Sequential(
            model3.conv1,
            model3.bn1,
            model3.relu,
            model3.maxpool,
            model3.layer1,
            model3.layer2,
            model3.layer3,
            model3.layer4,
            model3.avgpool)

        num_ftrs = model1.fc.in_features
        self.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x1 = self.features1(x[:, 0:3, :, :])
        x2 = self.features2(x[:, 3:6, :, :])
        x3 = self.features3(x[:, 6:9, :, :])
        x = torch.max(x1, x2)
        x = torch.max(x, x3)
        x = self.classifier(torch.squeeze(torch.squeeze(x,3),2))
        return x



class COVNet_trip_score(nn.Module):

    def __init__(self, model1,model2, model3, num_classes=2):
        super(COVNet_trip_score, self).__init__()

        self.features1 = nn.Sequential(
            model1.conv1,
            model1.bn1,
            model1.relu,
            model1.maxpool,
            model1.layer1,
            model1.layer2,
            model1.layer3,
            model1.layer4,
            model1.avgpool)


        self.features2 = nn.Sequential(
            model2.conv1,
            model2.bn1,
            model2.relu,
            model2.maxpool,
            model2.layer1,
            model2.layer2,
            model2.layer3,
            model2.layer4,
            model2.avgpool)


        self.features3 = nn.Sequential(
            model3.conv1,
            model3.bn1,
            model3.relu,
            model3.maxpool,
            model3.layer1,
            model3.layer2,
            model3.layer3,
            model3.layer4,
            model3.avgpool)

        num_ftrs = model1.fc.in_features
        self.classifier1 = nn.Linear(num_ftrs, num_classes)
        self.classifier2 = nn.Linear(num_ftrs, num_classes)
        self.classifier3 = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x1 = self.features1(x[:, 0:3, :, :])
        x1 = self.classifier1(torch.squeeze(torch.squeeze(x1,3),2))
        x2 = self.features2(x[:, 3:6, :, :])
        x2 = self.classifier2(torch.squeeze(torch.squeeze(x2,3),2))
        x3 = self.features3(x[:, 6:9, :, :])
        x3 = self.classifier3(torch.squeeze(torch.squeeze(x3,3),2))
        x = torch.mean(torch.stack((x1,x2,x3),dim=2),dim=2)

        return x

class COVNet_trip_scoread(nn.Module):

    def __init__(self, model1,model2, model3, num_classes=2):
        super(COVNet_trip_scoread, self).__init__()

        self.features1 = nn.Sequential(
            model1.conv1,
            model1.bn1,
            model1.relu,
            model1.maxpool,
            model1.layer1,
            model1.layer2,
            model1.layer3,
            model1.layer4,
            model1.avgpool)


        self.features2 = nn.Sequential(
            model2.conv1,
            model2.bn1,
            model2.relu,
            model2.maxpool,
            model2.layer1,
            model2.layer2,
            model2.layer3,
            model2.layer4,
            model2.avgpool)


        self.features3 = nn.Sequential(
            model3.conv1,
            model3.bn1,
            model3.relu,
            model3.maxpool,
            model3.layer1,
            model3.layer2,
            model3.layer3,
            model3.layer4,
            model3.avgpool)

        num_ftrs = model1.fc.in_features
        self.classifier1 = nn.Linear(num_ftrs, num_classes)
        self.classifier2 = nn.Linear(num_ftrs, num_classes)
        self.classifier3 = nn.Linear(num_ftrs, num_classes)
        self.weight = nn.Parameter(torch.tensor([1/3.0,1/3.0,1/3.0], requires_grad=True))

    def forward(self, x):
        x1 = self.features1(x[:, 0:3, :, :])
        x1 = self.classifier1(torch.squeeze(torch.squeeze(x1,3),2))
        x2 = self.features2(x[:, 3:6, :, :])
        x2 = self.classifier2(torch.squeeze(torch.squeeze(x2,3),2))
        x3 = self.features3(x[:, 6:9, :, :])
        x3 = self.classifier3(torch.squeeze(torch.squeeze(x3,3),2))
        #print(torch.sum(self.weight))
        #self.weight = self.weight/torch.sum(self.weight)
        self.weight.data = torch.max(self.weight.data,torch.zeros_like(self.weight.data))
        self.weight.data = self.weight.data / torch.sum(self.weight.data)
        x  = torch.matmul(torch.stack((x1,x2,x3),dim=2),self.weight)

        return x




class COVNet_trip_score_max(nn.Module):

    def __init__(self, model1,model2, model3, num_classes=2):
        super(COVNet_trip_score_max, self).__init__()

        self.features1 = nn.Sequential(
            model1.conv1,
            model1.bn1,
            model1.relu,
            model1.maxpool,
            model1.layer1,
            model1.layer2,
            model1.layer3,
            model1.layer4,
            model1.avgpool)


        self.features2 = nn.Sequential(
            model2.conv1,
            model2.bn1,
            model2.relu,
            model2.maxpool,
            model2.layer1,
            model2.layer2,
            model2.layer3,
            model2.layer4,
            model2.avgpool)


        self.features3 = nn.Sequential(
            model3.conv1,
            model3.bn1,
            model3.relu,
            model3.maxpool,
            model3.layer1,
            model3.layer2,
            model3.layer3,
            model3.layer4,
            model3.avgpool)

        num_ftrs = model1.fc.in_features
        self.classifier1 = nn.Linear(num_ftrs, num_classes)
        self.classifier2 = nn.Linear(num_ftrs, num_classes)
        self.classifier3 = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x1 = self.features1(x[:, 0:3, :, :])
        x1 = self.classifier1(torch.squeeze(torch.squeeze(x1,3),2))
        x2 = self.features2(x[:, 3:6, :, :])
        x2 = self.classifier2(torch.squeeze(torch.squeeze(x2,3),2))
        x3 = self.features3(x[:, 6:9, :, :])
        x3 = self.classifier3(torch.squeeze(torch.squeeze(x3,3),2))
        x = torch.max(x1, x2)
        x = torch.max(x, x3)
        return x




class COVNet_trip_mean(nn.Module):

    def __init__(self, model1,model2, model3, num_classes=2):
        super(COVNet_trip_mean, self).__init__()

        self.features1 = nn.Sequential(
            model1.conv1,
            model1.bn1,
            model1.relu,
            model1.maxpool,
            model1.layer1,
            model1.layer2,
            model1.layer3,
            model1.layer4,
            model1.avgpool)


        self.features2 = nn.Sequential(
            model2.conv1,
            model2.bn1,
            model2.relu,
            model2.maxpool,
            model2.layer1,
            model2.layer2,
            model2.layer3,
            model2.layer4,
            model2.avgpool)


        self.features3 = nn.Sequential(
            model3.conv1,
            model3.bn1,
            model3.relu,
            model3.maxpool,
            model3.layer1,
            model3.layer2,
            model3.layer3,
            model3.layer4,
            model3.avgpool)

        num_ftrs = model1.fc.in_features
        self.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x1 = self.features1(x[:, 0:3, :, :])
        x2 = self.features2(x[:, 3:6, :, :])
        x3 = self.features3(x[:, 6:9, :, :])
        x = torch.mean(torch.cat((x1,x2,x3),dim=2),dim=2,keepdim=True)

        x = self.classifier(torch.squeeze(torch.squeeze(x,3),2))
        return x




class COVNet_trip_cat(nn.Module):

    def __init__(self, model1,model2, model3, num_classes=2):
        super(COVNet_trip_cat, self).__init__()

        self.features1 = nn.Sequential(
            model1.conv1,
            model1.bn1,
            model1.relu,
            model1.maxpool,
            model1.layer1,
            model1.layer2,
            model1.layer3,
            model1.layer4,
            model1.avgpool)


        self.features2 = nn.Sequential(
            model2.conv1,
            model2.bn1,
            model2.relu,
            model2.maxpool,
            model2.layer1,
            model2.layer2,
            model2.layer3,
            model2.layer4,
            model2.avgpool)


        self.features3 = nn.Sequential(
            model3.conv1,
            model3.bn1,
            model3.relu,
            model3.maxpool,
            model3.layer1,
            model3.layer2,
            model3.layer3,
            model3.layer4,
            model3.avgpool)

        num_ftrs = model1.fc.in_features
        self.classifier = nn.Linear(3*num_ftrs, num_classes)

    def forward(self, x):
        x1 = self.features1(x[:, 0:3, :, :])
        x2 = self.features2(x[:, 3:6, :, :])
        x3 = self.features3(x[:, 6:9, :, :])
        x = torch.cat((x1,x2,x3),dim=1)

        x = self.classifier(torch.squeeze(torch.squeeze(x,3),2))
        return x




class COVNet_double_score(nn.Module):

    def __init__(self, model1,model2, num_classes=2):
        super(COVNet_double_score, self).__init__()

        self.features1 = nn.Sequential(
            model1.conv1,
            model1.bn1,
            model1.relu,
            model1.maxpool,
            model1.layer1,
            model1.layer2,
            model1.layer3,
            model1.layer4,
            model1.avgpool)


        self.features2 = nn.Sequential(
            model2.conv1,
            model2.bn1,
            model2.relu,
            model2.maxpool,
            model2.layer1,
            model2.layer2,
            model2.layer3,
            model2.layer4,
            model2.avgpool)



        num_ftrs = model1.fc.in_features
        self.classifier1 = nn.Linear(num_ftrs, num_classes)
        self.classifier2 = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x1 = self.features1(x[:, 6:9, :, :])
        x1 = self.classifier1(torch.squeeze(torch.squeeze(x1,3),2))
        x2 = self.features2(x[:, 3:6, :, :])
        x2 = self.classifier2(torch.squeeze(torch.squeeze(x2,3),2))
        x = torch.mean(torch.stack((x1,x2),dim=2),dim=2)

        return x


class COVNet_double_scoremax(nn.Module):

    def __init__(self, model1,model2, num_classes=2):
        super(COVNet_double_scoremax, self).__init__()

        self.features1 = nn.Sequential(
            model1.conv1,
            model1.bn1,
            model1.relu,
            model1.maxpool,
            model1.layer1,
            model1.layer2,
            model1.layer3,
            model1.layer4,
            model1.avgpool)
        self.features2 = nn.Sequential(
            model2.conv1,
            model2.bn1,
            model2.relu,
            model2.maxpool,
            model2.layer1,
            model2.layer2,
            model2.layer3,
            model2.layer4,
            model2.avgpool)
        num_ftrs = model1.fc.in_features
        self.classifier1 = nn.Linear(num_ftrs, num_classes)
        self.classifier2 = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x1 = self.features1(x[:, 6:9, :, :])
        x1 = self.classifier1(torch.squeeze(torch.squeeze(x1,3),2))
        x2 = self.features2(x[:, 3:6, :, :])
        x2 = self.classifier2(torch.squeeze(torch.squeeze(x2,3),2))
        x = torch.mean(torch.stack((x1,x2),dim=2),dim=2)

        return x

def resnet101_COVNet_double_score(num_classes=2, pretrained=True):
    model1 = models.resnet101(pretrained)
    model2 = models.resnet101(pretrained)

    return COVNet_double_score(model1,model2, num_classes)

def resnet101_COVNet_double_scoremax(num_classes=2, pretrained=True):
    model1 = models.resnet101(pretrained)
    model2 = models.resnet101(pretrained)

    return COVNet_double_scoremax(model1,model2, num_classes)

def resnet101_COVNet_trip_cat(num_classes=2, pretrained=True):
    model1 = models.resnet101(pretrained)
    model2 = models.resnet101(pretrained)
    model3 = models.resnet101(pretrained)

    return COVNet_trip_cat(model1,model2, model3, num_classes)

def resnet101_COVNet_trip_mean(num_classes=2, pretrained=True):
    model1 = models.resnet101(pretrained)
    model2 = models.resnet101(pretrained)
    model3 = models.resnet101(pretrained)

    return COVNet_trip_mean(model1,model2, model3, num_classes)
def resnet101_trip_score_max(num_classes=2, pretrained=True):
    model1 = models.resnet101(pretrained)
    model2 = models.resnet101(pretrained)
    model3 = models.resnet101(pretrained)

    return COVNet_trip_score_max(model1,model2, model3, num_classes)

def resnet101_COVNet_trip_score(num_classes=2, pretrained=True):
    model1 = models.resnet101(pretrained)
    model2 = models.resnet101(pretrained)
    model3 = models.resnet101(pretrained)

    return COVNet_trip_score(model1,model2, model3, num_classes)

def resnet101_COVNet_trip_scoread(num_classes=2, pretrained=True):
    model1 = models.resnet50(pretrained)
    model2 = models.resnet50(pretrained)
    model3 = models.resnet50(pretrained)
    return COVNet_trip_scoread(model1,model2, model3, num_classes)

def resnet101_COVNet_trip_max(num_classes=2, pretrained=True):
    model1 = models.resnet101(pretrained)
    model2 = models.resnet101(pretrained)
    model3 = models.resnet101(pretrained)

    return COVNet_trip_max(model1,model2, model3, num_classes)
def resnet101_COVNet(num_classes=2, pretrained=True):
    model = models.resnet101(pretrained)

    return COVNet(model, num_classes)

def resnet152_COVNet(num_classes=2, pretrained=True):
    model = models.resnet152(pretrained)

    return COVNet(model, num_classes)
def resnet152_COVNet_trip_max(num_classes=2, pretrained=True):
    model1 = models.resnet152(pretrained)
    model2 = models.resnet152(pretrained)
    model3 = models.resnet152(pretrained)

    return COVNet_trip_max(model1,model2, model3, num_classes)
def resnet152_COVNet_trip_score(num_classes=2, pretrained=True):
    model1 = models.resnet152(pretrained)
    model2 = models.resnet152(pretrained)
    model3 = models.resnet152(pretrained)

    return COVNet_trip_score(model1,model2, model3, num_classes)
def resnet152_trip_score_max(num_classes=2, pretrained=True):
    model1 = models.resnet152(pretrained)
    model2 = models.resnet152(pretrained)
    model3 = models.resnet152(pretrained)

    return COVNet_trip_score_max(model1,model2, model3, num_classes)
def resnet152_COVNet_trip_mean(num_classes=2, pretrained=True):
    model1 = models.resnet152(pretrained)
    model2 = models.resnet152(pretrained)
    model3 = models.resnet152(pretrained)

    return COVNet_trip_mean(model1,model2, model3, num_classes)
def resnet152_COVNet_trip_cat(num_classes=2, pretrained=True):
    model1 = models.resnet152(pretrained)
    model2 = models.resnet152(pretrained)
    model3 = models.resnet152(pretrained)

    return COVNet_trip_cat(model1,model2, model3, num_classes)

def resnet152_COVNet_double_score(num_classes=2, pretrained=True):
    model1 = models.resnet152(pretrained)
    model2 = models.resnet152(pretrained)

    return COVNet_double_score(model1,model2, num_classes)

def resnet152_COVNet_trip_scoread(num_classes=2, pretrained=True):
    model1 = models.resnet50(pretrained)
    model2 = models.resnet50(pretrained)
    model3 = models.resnet50(pretrained)
    return COVNet_trip_scoread(model1,model2, model3, num_classes)

def resnet152_COVNet_double_scoremax(num_classes=2, pretrained=True):
    model1 = models.resnet152(pretrained)
    model2 = models.resnet152(pretrained)

    return COVNet_double_scoremax(model1,model2, num_classes)


def resnet50_COVNet(num_classes=2, pretrained=True):
    model = models.resnet50(pretrained)

    return COVNet(model, num_classes)
def resnet50_COVNet_trip_max(num_classes=2, pretrained=True):
    model1 = models.resnet50(pretrained)
    model2 = models.resnet50(pretrained)
    model3 = models.resnet50(pretrained)

    return COVNet_trip_max(model1,model2, model3, num_classes)
def resnet50_COVNet_trip_score(num_classes=2, pretrained=True):
    model1 = models.resnet50(pretrained)
    model2 = models.resnet50(pretrained)
    model3 = models.resnet50(pretrained)
    return COVNet_trip_score(model1, model2, model3, num_classes)

def resnet50_COVNet_trip_scoread(num_classes=2, pretrained=True):
    model1 = models.resnet50(pretrained)
    model2 = models.resnet50(pretrained)
    model3 = models.resnet50(pretrained)

    return COVNet_trip_scoread(model1,model2, model3, num_classes)
def resnet50_trip_score_max(num_classes=2, pretrained=True):
    model1 = models.resnet50(pretrained)
    model2 = models.resnet50(pretrained)
    model3 = models.resnet50(pretrained)

    return COVNet_trip_score_max(model1,model2, model3, num_classes)
def resnet50_COVNet_trip_mean(num_classes=2, pretrained=True):
    model1 = models.resnet50(pretrained)
    model2 = models.resnet50(pretrained)
    model3 = models.resnet50(pretrained)

    return COVNet_trip_mean(model1,model2, model3, num_classes)
def resnet50_COVNet_trip_cat(num_classes=2, pretrained=True):
    model1 = models.resnet50(pretrained)
    model2 = models.resnet50(pretrained)
    model3 = models.resnet50(pretrained)

    return COVNet_trip_cat(model1,model2, model3, num_classes)

def resnet50_COVNet_double_score(num_classes=2, pretrained=True):
    model1 = models.resnet50(pretrained)
    model2 = models.resnet50(pretrained)

    return COVNet_double_score(model1,model2, num_classes)

def resnet50_COVNet_double_scoremax(num_classes=2, pretrained=True):
    model1 = models.resnet50(pretrained)
    model2 = models.resnet50(pretrained)

    return COVNet_double_scoremax(model1,model2, num_classes)

