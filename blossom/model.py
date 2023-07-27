from collections import OrderedDict
from torch import nn
from torchvision import models



architecture = {
    "MNASNet": models.mnasnet1_3(weights='IMAGENET1K_V1')  ,
    "Efficientnet" : models.efficientnet_b3(weights='IMAGENET1K_V1') ,
    "Swin_T": models.swin_t(weights='IMAGENET1K_V1'),
    "maxvit_t": models.maxvit_t(weights='IMAGENET1K_V1'),
    "ConvNeXt": models.convnext_tiny(weights='IMAGENET1K_V1'),
    "RegNet": models.regnet_y_3_2gf(weights='IMAGENET1K_V1')   
}

class ModifiedSqueezenet(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self, num_classes=102, trainable=False, model_name='Efficientnet'):
        super().__init__()
        self.model = architecture.get(model_name)
        self._freeze(trainable)
        self.model.model.num_classes = num_classes
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
        output = nn.Sequential(OrderedDict([('dropout1', nn.Dropout(p=0.55, inplace=True)),
                                            ('conv1', nn.Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1))),
                                            ('relu1', nn.ReLU(inplace=True)),
                                            ('pool', nn.MaxPool2d(kernel_size=(3, 3), stride=(
                                                1, 1), dilation=1, ceil_mode=True)),
                                            ('dropout2', nn.Dropout(p=0.5, inplace=True)),
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
