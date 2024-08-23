This is a repository for project "Integrating Pelican with Pytorch". 

Pelican Website: [https://pelicanplatform.org/](https://pelicanplatform.org/)

HTCondor: https://htcondor.org/

## What does this repo contains:

In Benchmark:

- `Benchmark1.ipynb`, `Benchmark2.ipynb`.

  Two Jupyter notebooks, contain two benchmark example with different datasets. Dataset information will be list in section [Dataset Using](## Dataset-using).

- `bm.py`

  A pytorch script version of benchmark2, allow you to pass arguments to choose different model, batch size, etc. See details in `README.md` inside Benchmark folder.

- `remote_image_folder.py`

  A custom class inherits `VisionDataset` of PyTorch, used for `Benchmark2.ipnb`

In doc:

- `UsingpyTorchwithPelican.md`

  Tutorial guides you through setting up and using PyTorch with Pelican for efficient data management and processing.

- `UsingpyTorchwithPelicanandHTCondor.md`

  An integrated tutorial for using pytorch with Pelican and HTCondor. 

In Others:

- `Test RemoteImageFolder.ipynb`

  Shows the using of RemoteImageFolder.

## Datasets:

[Fashion-mnist](https://www.kaggle.com/datasets/zalando-research/fashionmnist)

[ImageNet](https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/description)

[ImageNet Mini](https://github.com/fastai/imagenette)

## Current data under `chtc/PUBLIC/hzhao292` namespace



| SIZE | FILE                    |
| ---- | ----------------------- |
| 22M  | fashion-mnist_test.csv  |
| 5.4M | fashion-mnist_test.zip  |
| 127M | fashion-mnist_train.csv |
| 33M  | fashion-mnist_train.zip |
| 159G | ImageNet                |
| 156G | ImageNet.zip            |
| 1.5G | ImageNetMini            |
| 1.5G | ImageNetMini.zip        |
| 22G  | ImageNetSmall           |
| 21G  | ImageNetSmall.zip       |
| 114M | ImageNetTini            |
| 112M | ImageNetTini.zip        |
| 4.0K | test.txt                |

### Data for Benchmark1

| Size | File Name               |
| ---- | ----------------------- |
| 22M  | fashion-mnist_test.csv  |
| 5.4M | fashion-mnist_test.zip  |
| 127M | fashion-mnist_train.csv |
| 33M  | fashion-mnist_train.zip |

### Data for Benchmark2

| SIZE | FILE NAME        |
| ---- | ---------------- |
| 161G | ImageNet         |
| 156G | ImageNet.zip     |
| 1.5G | ImageNetMini.zip |
| 1.5G | ImageNetMini     |


For ImageNet standard data, train file is under `/train.` 

Under this path, there are 1000 directories named with corresponding classes of these images inside the directories. 

Val and Test follow the same rule.

ImageNetMini is a subset for convenient testing.  It only have 10 classes. About 1000 images in each class.
