import json

from model import ModifiedSqueezenet
from util import load_checkpoint, predict, label_name
from input_args import get_users_args


def main():
    user_arg = get_users_args()

    with open(user_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)

    squeezenet = ModifiedSqueezenet()
    squeezenet = load_checkpoint(user_arg.checkpoint_path, squeezenet)
    squeezenet.eval()
    _, top_p, top_class = predict(
        user_arg.input, squeezenet, topk=user_arg.top_k, device=user_arg.device)
    labels, top_p = label_name(top_p, top_class, idx_to_class=cat_to_name)
    print(f"The top {user_arg.top_k} probabilities are: {top_p}",
          f"The top {user_arg.top_k} classes are:{labels}")


if __name__ == "__main__":
    main()
