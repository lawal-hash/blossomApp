
# BlossomApp

AI algorithms are increasingly becoming integrated into various everyday applications. For instance, one might consider incorporating an image classifier into a smartphone app. To achieve this, a deep learning model, trained on a vast dataset of images, would be utilized as an essential component of the overall app architecture. In the future, a significant portion of software development will involve the incorporation of such models as standard elements in applications.

The current project involves training an image classifier designed to identify various species of flowers. This kind of classifier could be envisioned as a feature in a mobile app that provides the user with the name of the flower their camera captures. In practice, you would train this classifier and then export it for seamless integration into your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories.

## Run Locally

Clone the project

```bash
  git clone https://github.com/lawal-hash/blossomApp.git
```

Go to the project directory

```bash
  cd blossomApp
```

Install dependencies

```bash
  pip download requirements.txt
```



## Usage/Examples

```bash
  cd blossomApp
  python train.py --dir flowers/ --arch ResNet
```

```bash
  cd blossomApp
  python train.py --dir flowers/ --learning_rate 0.01  --epochs 20 --device cpy
```

```bash
  cd blossomApp
  python predict.py --input test_images/desert-rose.jpeg --checkpoint_path checkpoint/checkpoint_resnet_all.pth  --arch ResNet
```
## Contributing

Find any typos? Contributions are welcome!

First, fork this repository.

[![portfolio](https://raw.githubusercontent.com/udacity/ud777-writing-readmes/master/images/fork-icon.png)]()

Next, clone this repository to your desktop to make changes.

```
$ git clone {YOUR_REPOSITORY_CLONE_URL}

```

Once you've pushed changes to your local repository, you can issue a pull request by clicking on the green pull request icon.

[![portfolio](https://raw.githubusercontent.com/udacity/ud777-writing-readmes/master/images/pull-request-icon.png)]()





## Appendix

Any additional information goes here

[![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/Ayan_Yemi)
