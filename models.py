import torch
import torchvision
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import model_urls
from torchvision.models.resnet import ResNet, Bottleneck


class ResNetShort(ResNet):
    """
    Shortened ResNet originally designed by Seongwoon
    """
    def override_layers(self):
        self.avgpool = nn.MaxPool2d(kernel_size=4, stride=4, padding=0) # Seongwoon
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(9216, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
        )


class SkipNet(ResNet):
    """
    SkipNet
    Pool features from every layer
    """

    def override_layers(self):
        self.pool1 = nn.MaxPool2d(kernel_size=56, stride=1, padding=0) # Pools end of layer1
        self.pool2 = nn.MaxPool2d(kernel_size=14, stride=7, padding=0) # Pools end of layer2
        self.pool3 = nn.MaxPool2d(kernel_size=4, stride=4, padding=0) # Seongwoon
        self.pool4 = nn.AvgPool2d(7, stride=1) # ResNet 152 standard

        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(14080, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        f1 = self.pool1(x1)
        f2 = self.pool2(x2)
        f3 = self.pool3(x3)
        #f4 = self.pool4(x4)

        f1 = f1.view(f1.size(0),-1)
        f2 = f2.view(f2.size(0),-1)
        f3 = f3.view(f3.size(0),-1)
        #f4 = f4.view(f4.size(0),-1)

        #print('f1 size:',f1.size())
        #print('f2.size:',f2.size())
        #print('f3.size:',f3.size())
        #print('f4.size:',f4.size())

        x = torch.cat([f1,f2,f3],1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



def resnet_short(pretrained=False, **kwargs):
    """Constructs a short resnet model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetShort(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model_urls = torchvision.models.resnet.model_urls
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    model.override_layers()
    return model



def skip_net(pretrained=False, **kwargs):
    """Constructs a ResSkipNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SkipNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model_urls = torchvision.models.resnet.model_urls
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    model.override_layers()
    return model
