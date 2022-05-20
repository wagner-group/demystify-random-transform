'''Simple fully-connected models'''


import torch
import torch.nn as nn

from .base_model import BaseModel


class DenseModel(BaseModel):

    def __init__(self, input_dim, num_classes=2):
        super(DenseModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 400)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(400, 400)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(400, num_classes)

    def forward(self, x, **kwargs):
        if x.dim() > 2:
            x = x.squeeze()
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


class DenseModelV2(BaseModel):

    def __init__(self, input_dim, num_classes=2):
        super(DenseModelV2, self).__init__()
        self.fc1 = nn.Linear(input_dim, 800)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(800, 800)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(800, 800)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(800, 800)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(800, num_classes)

    def forward(self, x, **kwargs):
        if x.dim() > 2:
            x = x.squeeze()
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        x = self.fc5(x)
        return x


class DenseModelV3(BaseModel):

    def __init__(self, input_dim, num_classes=2):
        super(DenseModelV3, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2000)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(2000, 2000)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(2000, 2000)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(2000, 2000)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(2000, 2000)
        self.relu5 = nn.ReLU(inplace=True)
        self.fc6 = nn.Linear(2000, 2000)
        self.relu6 = nn.ReLU(inplace=True)
        self.fc7 = nn.Linear(2000, 2000)
        self.relu7 = nn.ReLU(inplace=True)
        self.fc8 = nn.Linear(2000, 2000)
        self.relu8 = nn.ReLU(inplace=True)
        self.fc9 = nn.Linear(2000, 400)
        self.relu9 = nn.ReLU(inplace=True)
        self.fc10 = nn.Linear(400, num_classes)

    def forward(self, x, **kwargs):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        x = self.relu5(self.fc5(x))
        x = self.relu6(self.fc6(x))
        x = self.relu7(self.fc7(x))
        x = self.relu8(self.fc8(x))
        x = self.relu9(self.fc9(x))
        x = self.fc10(x)
        return x


class LinearModel(BaseModel):

    def __init__(self, input_dim, num_classes=2):
        super(LinearModel, self).__init__()
        if num_classes > 1:
            self.fc1 = nn.Linear(input_dim, num_classes)
        else:
            self.fc1 = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        return x


class AffineTransform(BaseModel):
    """
    A simple module that involves two networks: one for parameterizing an affine
    transformation (i.e. scale and shift) and the other for classification.
    """

    def __init__(self, transformer_net, classify_net):
        super(AffineTransform, self).__init__()
        self.transformer_net = transformer_net
        self.classify_net = classify_net

    def forward(self, x, **kwargs):
        output = self.transformer_net(x)
        if isinstance(output, tuple):
            x = output[0] * x + output[1]
        else:
            x = output * x

        return self.classify_net(x)


class ScaleNetwork(BaseModel):
    """Network for parameterizing a scaling function"""

    def __init__(self, input_dim):
        super(ScaleNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2000)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(2000, 2000)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(2000, 2000)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(2000, 400)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(400, 1)
        # We want to initialize scale to 1
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.constant_(m.weight, 0)
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, x, **kwargs):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        x = self.fc5(x).exp()
        # x = self.fc5(x).sigmoid() * 5
        return x


class ElemAffineNetwork(BaseModel):
    """Network for parameterizing affine transformation"""

    def __init__(self, input_dim):
        super(ElemAffineNetwork, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 2000)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(2000, 2000)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(2000, 2000)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(2000, 2000)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(2000, 2 * input_dim)
        # We want to initialize scale to 1 and shift to 0
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)

    def forward(self, x, **kwargs):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        x = self.fc5(x)
        # Use exp to make sure that scale is a positive number
        scale = torch.exp(x[:, :self.input_dim // 2])
        # Use tanh to keep shift in [-1, 1]
        shift = torch.tanh(x[:, self.input_dim // 2:])
        return scale, shift


class AlexNet(BaseModel):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x, **kwars):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x


class BasicModel(BaseModel):

    def __init__(self, num_classes=10):
        super(BasicModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=3)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(10368, 1024)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024, 128)
        self.relu5 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(128, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, **kwars):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.view(x.size(0), -1)
        x = self.relu4(self.fc1(x))
        x = self.relu5(self.fc2(x))
        x = self.fc3(x)
        return x
