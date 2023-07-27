import json

from blossom.model import BlossomNet
from blossom.util import predict, label_name, load_checkpoint

from blossom.input_args import get_users_args


def main():
    user_arg = get_users_args()

    with open(user_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)

    blossomnet = BlossomNet()
    blossomnet = load_checkpoint(user_arg.checkpoint_path, blossomnet)
    blossomnet.eval()
    _, top_p, top_class = predict(
        user_arg.input, blossomnet,user_arg.arch, top_k=user_arg.top_k, device=user_arg.device)
    labels, top_p = label_name(top_p, top_class, idx_to_class=cat_to_name)
    print(f"The top {user_arg.top_k} probabilities are: {top_p}",
        f"The top {user_arg.top_k} classes are:{labels}")


if __name__ == "__main__":
    main()
