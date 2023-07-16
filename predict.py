import json
import os

from torch import nn, optim
from torchvision import transforms

from model import ModifiedSqueezenet
from util import load_checkpoint, predict, label_name



def main():
    user_arg = get_users_args()
    
    with open(user_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    squeezenet = ModifiedSqueezenet()
    squeezenet = load_checkpoint(user_arg.checkpoint_path, squeezenet)
    image, top_p, top_class = predict(user_arg.input, squeezenet, topk=user_arg.top_k, device=user_arg.device)
    

    
