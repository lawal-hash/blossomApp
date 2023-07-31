from collections import OrderedDict
from torch import nn
from torchvision.models import get_model

architecture = {
    "MNASNet": 'mnasnet1_3',
    "Efficientnet": 'efficientnet_b3',
    # "Swin_T": 'swin_t',
    #"maxvit_t": 'maxvit_t',
    "ConvNeXt": 'convnext_tiny',
    "RegNet": 'regnet_y_3_2gf'
}


HIDDEN_INPUT = {
    "MNASNet": 20480,
    "Efficientnet": 24576,
    # "Swin_T": ,
    "maxvit_t": 512,
    "ConvNeXt": 12288,
    "RegNet":  1512
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
        self.num_classes = num_classes
        self.model_name = model_name
        
        if hasattr(self.model, 'avgpool'):
            self.model.avgpool = nn.AvgPool2d((3, 3), stride=(2, 2))
            
        if hasattr(self.model, 'classifier') and (self.model_name == 'ConvNeXt'):
            self.model.classifier = self.convnext_output()
            
        if hasattr(self.model, 'classifier') and (self.model_name == 'Efficientnet'):
            self.model.classifier = self.efficientnet_output()    
            
            
        if hasattr(self.model, 'classifier') and (self.model_name == 'MNASNet'):
            self.model.classifier = self.mnasnet_output()          
            
        #elif hasattr(self.model, 'linear'):
            #self.model.linear = self._output()
        #elif hasattr(self.model, 'fc'):
            #self.model.fc = self._output()
        #else:
            #raise AttributeError('Unsupported architecture')



    def _freeze(self, trainable):
        """_summary_

        Args:
            trainable (_type_): _description_
        """
        for param in self.model.parameters():
            param.requires_grad = trainable

    def efficientnet_output(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        output = nn.Sequential(OrderedDict([ ('flatten', nn.Flatten()),
                                            ('dropout1', nn.Dropout(p=0.5, inplace=False)),
                                            ('linear1', nn.Linear(HIDDEN_INPUT.get(self.model_name), 3650)),
                                            ('relu1', nn.ReLU(inplace=False)),
                                            ('dropout2', nn.Dropout(p=0.5, inplace=False)),
                                            ('linear2', nn.Linear(3650, self.num_classes)),
                                            ('output', nn.LogSoftmax(dim=1))
                                            ]))
        return output
    
    def mnasnet_output(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        output = nn.Sequential(OrderedDict([
                                        ('pool',nn.AvgPool2d((3, 3), stride=(2, 2))),
                                        ('flatten', nn.Flatten()),
                                        ('dropout1', nn.Dropout(p=0.5, inplace=False)),
                                        ('linear1', nn.Linear(HIDDEN_INPUT.get(self.model_name), 512)),
                                        ('relu1', nn.ReLU(inplace=False)),
                                        ('dropout2', nn.Dropout(p=0.5, inplace=False)),
                                        ('linear2', nn.Linear(512, self.num_classes)),
                                        ('output', nn.LogSoftmax(dim=1))
                                        ]))
        return output





    def convnext_output(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        output = nn.Sequential(OrderedDict([
                                    ('flatten', nn.Flatten()),
                                    ('dropout1', nn.Dropout(p=0.5, inplace=False)),
                                    ('linear1', nn.Linear(HIDDEN_INPUT.get(self.model_name), 3250)),
                                    ('relu1', nn.ReLU(inplace=False)),
                                    ('dropout2', nn.Dropout(p=0.3, inplace=False)),
                                    ('linear2', nn.Linear(3250, self.num_classes)),
                                    ('output', nn.LogSoftmax(dim=1))
                                    ]))
        return output


    def _output(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        output = nn.Sequential(OrderedDict([('dropout1', nn.Dropout(p=0.3, inplace=False)),
                                        ('linear1', nn.Linear(HIDDEN_INPUT.get(self.model_name), 365)),
                                        ('relu1', nn.ReLU(inplace=False)),
                                        ('linear2', nn.Linear(365, self.num_classes)),
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
        return  self.model(input_x)
