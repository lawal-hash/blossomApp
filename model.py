from collections import OrderedDict
from torch import nn
from torchvision import models


class ModifiedSqueezenet(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self, num_classes=102, trainable=True):
        super().__init__()
        self.model = models.squeezenet1_1(pretrained=True)
        self._freeze(trainable)
        self.model.num_classes = num_classes
        self.model.classifier = self._output()

    def _freeze(self, trainable):
        """_summary_

        Args:
            trainable (_type_): _description_
        """
        for param in self.model.parameters():
            param.requires_grad = trainable

    def _output(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        output = nn.Sequential(OrderedDict([('conv1', nn.Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1))),
                                            ('relu1', nn.ReLU(inplace=True)),
                                            ('pool', nn.MaxPool2d(kernel_size=(3, 3), stride=(
                                                1, 1), dilation=1, ceil_mode=True)),
                                            ('conv2', nn.Conv2d(
                                                128, 102, kernel_size=(3, 3), stride=(1, 1))),
                                            ('relu2', nn.ReLU(inplace=True)),
                                            ('global_avgpool', nn.AvgPool2d(
                                                kernel_size=7, stride=1, padding=0)),
                                            ('output', nn.LogSoftmax(dim=1))
                                            ]))
        return output

    def forward(self, input_x):
        """_summary_

        Args:
            input_x (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.model(input_x)
