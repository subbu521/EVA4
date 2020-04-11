from tqdm import tqdm_notebook, tnrange
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

dropout_value = 0.07

class newResnet(nn.Module):
    def __init__(self):
        super(newResnet, self).__init__()

        self.PrepLayer = nn.Sequential(
          nn.Conv2d(3, 64, 3,padding=1),
          nn.ReLU(),
          nn.BatchNorm2d(64),
          nn.Dropout(dropout_value)      
        )

        self.layer1convblock1 = nn.Sequential(
          nn.Conv2d(64, 128, 3,padding=1),
          nn.MaxPool2d(2, 2),
          nn.ReLU(),
          nn.BatchNorm2d(128),
          nn.Dropout(dropout_value)      
        )

        self.resnet1 = nn.Sequential(
          nn.Conv2d(128, 128, 3,padding=1),
          nn.BatchNorm2d(128),
          nn.ReLU(),
          nn.Conv2d(128, 128, 3,padding=1),
          nn.BatchNorm2d(128),
          nn.ReLU(),
          nn.Dropout(dropout_value) 
        )

        self.layer2convblock2 = nn.Sequential(
            nn.Conv2d(128, 256, 3,padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(dropout_value)      
        )
       
        self.layer3convblock3 = nn.Sequential(
          nn.Conv2d(256, 512, 3,padding=1),
          nn.MaxPool2d(2, 2),
          nn.ReLU(),
          nn.BatchNorm2d(512),
          nn.Dropout(dropout_value)      
        )

        self.resnet3 = nn.Sequential(
          nn.Conv2d(512, 512, 3,padding=1),
          nn.BatchNorm2d(512),
          nn.ReLU(),
          nn.Conv2d(512, 512, 3,padding=1),
          nn.BatchNorm2d(512),
          nn.ReLU(),
          nn.Dropout(dropout_value) 
        )
        
        self.pool1 = nn.MaxPool2d(4, 1)


        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        ) 


    def forward(self, x):
      
        x = self.PrepLayer(x)

        x = self.layer1convblock1(x)
        r1 = self.resnet1(x)
        x = torch.add(x, r1)

        x = self.layer2convblock2(x)

        x = self.layer3convblock3(x)
        r3 = self.resnet3(x)
        x = torch.add(x, r3)

        x = self.pool1(x) 
  
        x = self.fc(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)











dropout_value = 0.07
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.convblock1 = nn.Sequential(
          nn.Conv2d(3, 32, 3,padding=1),
          nn.ReLU(),
          nn.BatchNorm2d(32),
          nn.Dropout(dropout_value)
        )

        self.convblock2 = nn.Sequential(
          nn.Conv2d(32, 32, 3,padding=1),
          nn.ReLU(),
          nn.BatchNorm2d(32),
          nn.Dropout(dropout_value)
        )

        self.convblock3 = nn.Sequential(
          nn.Conv2d(32, 32, 3,padding=1),
          nn.ReLU(),
          nn.BatchNorm2d(32),
          nn.Dropout(dropout_value)      
        )

        self.pool1 = nn.MaxPool2d(2, 2)

        self.convblock4 = nn.Sequential(
	      nn.Conv2d(32, 64, 3,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Dropout(dropout_value)
        )

        self.convblock5 = nn.Sequential(
        nn.Conv2d(64, 64, 3,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Dropout(dropout_value)
        )

     

        self.pool2 = nn.MaxPool2d(2, 2)

        self.dconvblock1 = nn.Sequential(
	      nn.Conv2d(64, 128, 3,dilation=2,padding=2),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.Dropout(dropout_value)
        )

        self.convblock6 = nn.Sequential(
        nn.Conv2d(64, 128, 3,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.Dropout(dropout_value)
        )

        self.convblock7 = nn.Sequential(
        nn.Conv2d(128, 128, 3,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.Dropout(dropout_value)      
        )

        self.pool3 = nn.MaxPool2d(2, 2)

        self.convblock8 = nn.Sequential(
        nn.Conv2d(128, 128, 3,groups=128,dilation=1,padding=1),
        nn.Conv2d(128, 256, kernel_size=(1,1))
        
        
          
        )

        self.convblock9 = nn.Sequential(
        nn.Conv2d(256, 256, 3,groups=256,dilation=1,padding=1),
        nn.Conv2d(256, 256, kernel_size=(1,1)),
           
        )


        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=(4,4))
        ) # output_size = 1

        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        ) 




    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.pool2(x)
        x1 = self.dconvblock1(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = torch.add(x, x1)

        x = self.pool3(x)
        x = self.convblock8(x)
        x = self.convblock9(x)

        x = self.gap(x)        
        x = self.convblock10(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())