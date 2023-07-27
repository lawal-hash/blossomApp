import os

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from blossom.model import BlossomNet
from blossom.trainer import Trainer
from blossom.input_args import get_users_args
from blossom.transform import RESIZE, CROP


def main():
    user_arg = get_users_args()

    train_transform = transforms.Compose([
        transforms.Resize(RESIZE.get(user_arg.arch)),
        transforms.CenterCrop(CROP.get(user_arg.arch)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(RESIZE.get(user_arg.arch)),
        transforms.CenterCrop(CROP.get(user_arg.arch)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])
    train_dir = os.path.join(user_arg.data_directory, 'train')
    valid_dir = os.path.join(user_arg.data_directory, 'valid')
    test_dir = os.path.join(user_arg.data_directory, 'test')

    train_datasets = ImageFolder(train_dir, transform=train_transform)
    valid_datasets = ImageFolder(valid_dir, transform=test_transform)
    test_datasets = ImageFolder(test_dir, transform=test_transform)

    train_dataloader = DataLoader(
        train_datasets, batch_size=user_arg.train_batchsize, shuffle=True)
    valid_dataloader = DataLoader(
        valid_datasets, batch_size=user_arg.test_batchsize, shuffle=True)
    test_dataloader = DataLoader(
        test_datasets, batch_size=user_arg.test_batchsize, shuffle=True)

    blossomnet = BlossomNet(
        num_classes=user_arg.num_classes, trainable=user_arg.trainable, model_name=user_arg.arch)
    trainer = Trainer(model=blossomnet, optimizer=optim.Adam(blossomnet.parameters(
    ), lr=user_arg.learning_rate), criterion=nn.NLLLoss(), device=user_arg.device, checkpoint=user_arg.checkpoint, path=user_arg.checkpoint_path)

    trainer.fit(train_dataloader, epochs=user_arg.epochs,
                validation_data=valid_dataloader)


if __name__ == "__main__":
    main()
