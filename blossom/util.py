from PIL import Image

import numpy as np
import torch
from torchvision.transforms import functional as F


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image_path)
    img_resized = F.resize(img, 256)
    cropped_img = F.center_crop(img_resized, 224)
    image = np.array(cropped_img)
    normalized_image = (
        image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    #tensor_img = F.pil_to_tensor(cropped_img)
    #tensor_img = F.normalize(tensor_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    tensor_img = torch.from_numpy(normalized_image.transpose(), )
    return tensor_img, img


def predict(image_path, model, topk=5, device = 'cuda'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)
    tensor_image, img = process_image(image_path)
    tensor_image = tensor_image.type('torch.FloatTensor')
    tensor_image = tensor_image.reshape(-1, 3, 224, 224)

    log_prob = model(tensor_image)
    prob = torch.exp(log_prob)
    top_p, top_class = prob.topk(topk, dim=1)
    return img, top_p, top_class


def label_name(top_p, top_class, idx_to_class=None):
    labels = []
    top_class = top_class.detach().numpy().reshape(-1,).tolist()
    top_p = top_p.detach().numpy().reshape(-1,).tolist()
    for target in top_class:
        labels.append(idx_to_class.get(str(target)))
    return labels, top_p


def load_checkpoint(path, model):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.num_classes = checkpoint['num_classes']
    return model