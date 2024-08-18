This is a repository for project "Integrating Pelican with Pytorch". It's still under constuction. The goal:

- Perform benchmark tests to validate model performance, ensuring effective integration of machine learning datasets through Pelican origins for improved data accessibility and processing.
- Develop and test models from small prototypes to extensive systems within an HTCondor-managed workflow, significantly optimizing computational throughput.
- Create comprehensive tutorials and practical guides to facilitate the adoption of PyTorch and Pelican integration across the machine learning community.

Pelican Website: https://pelicanplatform.org/

Related tools:

- HTCondor: https://htcondor.org/

## Datasets:

[Fashion-mnist](https://www.kaggle.com/datasets/zalando-research/fashionmnist)

[ImageNet](https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/description)

[ImageNet Mini](https://github.com/fastai/imagenette)

## Current data under `chtc/PUBLIC/hzhao292` namespace

| Size | File                    |
| ---- | ----------------------- |
| 22M  | fashion-mnist_test.csv  |
| 5.4M | fashion-mnist_test.zip  |
| 127M | fashion-mnist_train.csv |
| 33M  | fashion-mnist_train.zip |
| 159G | ImageNet                |
| 1.5G | ImageNetMini            |
| 1.5G | ImageNetMini.tgz        |
| 1.5G | ImageNetMini.zip        |
| 22G  | ImageNetSmall           |
| 21G  | ImageNetSmall.zip       |
| 114M | ImageNetTini            |
| 112M | ImageNetTini.zip        |
| 156G | ImageNet.zip            |
| 4.0K | test.txt                |

### Data for Benchmarking1

| Size | File Name               |
| ---- | ----------------------- |
| 22M  | fashion-mnist_test.csv  |
| 5.4M | fashion-mnist_test.zip  |
| 127M | fashion-mnist_train.csv |
| 33M  | fashion-mnist_train.zip |

### Data for Benchmarking2

| Size | File Name        |
| ---- | ---------------- |
| 161G | ImageNet         |
| 156G | ImageNet.zip     |
| 1.5G | ImageNetMini.tgz |
| 1.5G | ImageNetMini     |

For ImageNet standard data, the training files are under /train. 

Under this path, there is 1000 directories named with corresponding classes of these images inside the directories. 

Val and Test follow the same rule after preparation. 

ImageNetMini is a subset for convenient testing.  It only has 10 classes with 1000 images in each class.



## Newest Update

