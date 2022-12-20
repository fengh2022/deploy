import torch
import torch.nn as nn
from torchvision.models import resnet18




class ResNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(ResNet, self).__init__()
        self.model = resnet18(pretrained=pretrained)
        fc_features = self.fc.in_features
        self.model.fc = nn.Linear(fc_features, num_classes)

    def forward(self, x):
        return self.model(x)

def Conv2D(in_d, out_d, kernel_size, stride, padding):
    conv = nn.Conv2d(in_d, out_d, kernel_size=kernel_size, stride=stride, padding=padding)
    torch.nn.init.xavier_uniform_(conv.weight)
    return nn.Sequential(
        conv,
        nn.BatchNorm2d(out_d),
        # nn.ReLU(inplace=True),
        nn.LeakyReLU(inplace=True)
    )

class VGG(nn.Module):
    def __init__(self, num_classes=2):
        super(VGG, self).__init__()
        self.stage1 = nn.Sequential(
            Conv2D(3, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.stage2 = nn.Sequential(
            Conv2D(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.stage3 = nn.Sequential(
            Conv2D(128, 256, kernel_size=3, stride=1, padding=1),
            Conv2D(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.stage4 = nn.Sequential(
            Conv2D(256, 512, kernel_size=3, stride=1, padding=1),
            Conv2D(512, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.pooling = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(512, num_classes)
        self.out = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.pooling(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.out(x)

        return x


def get_model(model_name, num_classes):
    if model_name=='VGG':
        return VGG(num_classes=num_classes)
    elif model_name=='ResNet':
        return ResNet(num_classes=num_classes, pretrained=False)