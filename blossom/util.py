import json
from PIL import Image
import torch
from torchvision import transforms
from blossom.transform import RESIZE, CROP


def process_image(image_path, model_name):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image_path)
    test_transform = transforms.Compose([
        transforms.Resize(RESIZE.get(model_name)),
        transforms.CenterCrop(CROP.get(model_name)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    tensor_img = test_transform(img)
    return tensor_img, img


def predict(image_path, model, model_name, top_k=5, device='cuda'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)
    tensor_image, img = process_image(image_path, model_name)
    tensor_image = tensor_image.type('torch.FloatTensor')
    tensor_image = tensor_image.reshape(-1, 3,
                                        CROP.get(model_name)[0], CROP.get(model_name)[0])

    log_prob = model(tensor_image)
    prob = log_prob.exp()
    top_p, top_class = prob.topk(top_k, dim=1)
    return img, top_p, top_class


def label_name(top_p, top_class):

    with open('labels/cat_to_name.json', 'r') as file:
        cat_to_name = json.load(file)

    with open('labels/class_to_idx.json', 'r') as file:
        class_to_idx = json.load(file)
    labels = []
    top_class = top_class.detach().numpy().reshape(-1,).tolist()
    top_p = top_p.detach().numpy().reshape(-1,).tolist()

    keys = list(class_to_idx.keys())
    values = list(class_to_idx.values())
    for target in top_class:
        idx = values.index(target)
        class_name = cat_to_name.get(str(keys[idx]))
        labels.append(class_name)
    return labels, top_p


def load_checkpoint(path, model):
    checkpoint = torch.load(path,   map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.num_classes = checkpoint['num_classes']
    return model
