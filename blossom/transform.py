RESIZE = {
    "MNASNet": [232],
    "Efficientnet": [320],
    "Swin_T": [232],
    "maxvit_t": [224],
    "ConvNeXt": [236],
    "RegNet": [232]
}

CROP = {
    "MNASNet": [224],
    "Efficientnet": [300],
    "Swin_T": [224],
    "maxvit_t": [224],
    "ConvNeXt": [224],
    "RegNet":  [224]
}

HIDDEN_INPUT = {
    "MNASNet": 1280,
    "Efficientnet": 1536,
    # "Swin_T": ,
    "maxvit_t": 512,
    "ConvNeXt": 768,
    "RegNet":  1512
}
