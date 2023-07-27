import os

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from blossom.model import ModifiedSqueezenet
from blossom.trainer import Trainer
from blossom.input_args import get_users_args


def main():
    user_arg = get_users_args()

    train_transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        # transforms.Resize(256),
        transforms.CenterCrop(224),
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

    squeezenet = ModifiedSqueezenet(
        num_classes=user_arg.num_classes, trainable=user_arg.trainable)
    trainer = Trainer(model=squeezenet, optimizer=optim.Adam(squeezenet.parameters(
    ), lr=user_arg.learning_rate), criterion=nn.NLLLoss(), device=user_arg.device, checkpoint=user_arg.checkpoint, path=user_arg.checkpoint_path)

    trainer.fit(train_dataloader, epochs=user_arg.epochs,
                validation_data=valid_dataloader)


if __name__ == "__main__":
    main()
