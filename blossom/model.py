from collections import OrderedDict
from torch import nn
from torchvision.models import get_model

architecture = {
    "MNASNet": 'mnasnet1_3',
    "Efficientnet": 'efficientnet_b3',
    # "Swin_T": 'swin_t',
    "maxvit_t": 'maxvit_t',
    "ConvNeXt": 'convnext_tiny',
    "RegNet": 'regnet_y_3_2gf'
}


class BlossomNet(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, num_classes=102, trainable=False, model_name='Efficientnet'):
        super().__init__()
        self.model = get_model(architecture.get(model_name), weights='DEFAULT')
        self._freeze(trainable)
        self.model.model.num_classes = num_classes
        if hasattr(self.model, 'classifier'):
            self.model.classifier = self._output()
        elif hasattr(self.model, 'linear'):
            self.model.linear = self._output()
        elif hasattr(self.model, 'fc'):
            self.model.fc = self._output()
        else:
            raise AttributeError('Unsupported architecture')

        if hasattr(self.model, 'avgpool'):
            self.model.avgpool = nn.AdaptiveAvgPool2d(output_size=1)

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
                                            ('linear1', nn.Linear()),
                                            ('relu1', nn.ReLU(inplace=True)),
                                            ('linear2', nn.Linear()),
                                            ('relu2', nn.ReLU(inplace=True)),
                                            ('output', nn.LogSoftmax(dim=1))
                                            ]))
        return output

    def _linear_output(self):
        output = nn.Sequential(OrderedDict([('dropout1', nn.Dropout(p=0.55, inplace=True)),
                                            ('linear1', nn.Linear()),
                                            ('relu1', nn.ReLU(inplace=True)),
                                            ('linear2', nn.Linear()),
                                            ('relu2', nn.ReLU(inplace=True)),
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
