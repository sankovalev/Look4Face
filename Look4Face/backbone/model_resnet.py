import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, ReLU, Dropout, MaxPool2d, Sequential, Module


# Support: ['ResNet_50', 'ResNet_101', 'ResNet_152']

# свертка 3x3
def conv3x3(in_planes, out_planes, stride = 1):
    """3x3 convolution with padding"""

    return Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride,
                     padding = 1, bias = False)

# свертка 1x1
def conv1x1(in_planes, out_planes, stride = 1):
    """1x1 convolution"""

    return Conv2d(in_planes, out_planes, kernel_size = 1, stride = stride, bias = False)


class BasicBlock(Module):
    '''
    Основной блок сетки (residual block)
    '''
    expansion = 1 #расширение

    def __init__(self, inplanes, planes, stride = 1, downsample = None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)  #свертка 3x3
        self.bn1 = BatchNorm2d(planes)                  #батч норм.
        self.relu = ReLU(inplace = True)                #активация ReLU
        self.conv2 = conv3x3(planes, planes)            #свертка 3x3
        self.bn2 = BatchNorm2d(planes)                  #батч норм.
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        '''
        Прямой проход через residual block
        :param x: вход
        :return: выход
        '''
        identity = x #начальный вход

        out = self.conv1(x)   #свертка 3x3
        out = self.bn1(out)   #батч норм.
        out = self.relu(out)  #ReLU

        out = self.conv2(out) #свертка 3x3
        out = self.bn2(out)   #батч норм.

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity       #прибавляем в выходу блока его вход. Должны быть одного размера!
        out = self.relu(out)  #ReLU

        return out


class Bottleneck(Module):
    '''
    Блок снижения размерности
    '''
    expansion = 4 #расширение

    def __init__(self, inplanes, planes, stride = 1, downsample = None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)                  #свертка 1x1
        self.bn1 = BatchNorm2d(planes)                          #батч норм.
        self.conv2 = conv3x3(planes, planes, stride)            #свертка 3x3
        self.bn2 = BatchNorm2d(planes)                          #батч норм.
        self.conv3 = conv1x1(planes, planes * self.expansion)   #свертка 1x1 с расширением (увеличили размерность)
        self.bn3 = BatchNorm2d(planes * self.expansion)         #батч норм. расширенного тензора
        self.relu = ReLU(inplace = True)                        #активация ReLU
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        '''
        Прямой проход через блок
        :param x: вход
        :return: выход
        '''
        identity = x #начальный вход

        out = self.conv1(x)     #свертка 1x1
        out = self.bn1(out)     #батч норм.
        out = self.relu(out)    #активация ReLU

        out = self.conv2(out)   #свертка 3x3
        out = self.bn2(out)     #батч норм.
        out = self.relu(out)    #активация ReLU

        out = self.conv3(out)   #свертка 1x1 с расширением (увеличили размерность)
        out = self.bn3(out)     #батч норм. расширенного тензора

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity         #прибавляем в выходу блока его вход. Должны быть одного размера!
        out = self.relu(out)    #активация ReLU

        return out


#РАЗОБРАТЬ ЭТОТ КЛАСС
class ResNet(Module):
    '''
    ResNet-сетка
    '''

    def __init__(self, input_size, block, layers, zero_init_residual = True):
        super(ResNet, self).__init__()
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        self.inplanes = 64  #входное число слоев
        self.conv1 = Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False) #сверточный слой
        self.bn1 = BatchNorm2d(64) #батч норм.
        self.relu = ReLU(inplace = True) #активация ReLU
        self.maxpool = MaxPool2d(kernel_size = 3, stride = 2, padding = 1) #макс пулинг 3x3
        self.layer1 = self._make_layer(block, 64, layers[0]) #
        self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride = 2)

        self.bn_o1 = BatchNorm2d(2048)
        self.dropout = Dropout()
        if input_size[0] == 112:
            self.fc = Linear(2048 * 4 * 4, 512)
        else:
            self.fc = Linear(2048 * 8 * 8, 512)
        self.bn_o2 = BatchNorm1d(512)

        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn_o1(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn_o2(x)

        return x


def ResNet_50(input_size, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(input_size, Bottleneck, [3, 4, 6, 3], **kwargs)

    return model


def ResNet_101(input_size, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(input_size, Bottleneck, [3, 4, 23, 3], **kwargs)

    return model


def ResNet_152(input_size, **kwargs):
    """Constructs a ResNet-152 model.
    """
    model = ResNet(input_size, Bottleneck, [3, 8, 36, 3], **kwargs)

    return model
