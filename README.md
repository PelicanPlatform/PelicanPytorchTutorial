This is a repository for project "Integrating Pelican with Pytorch". It's still under constuction. Finally, we hope:

- Perform benchmark tests to validate model performance, ensuring effective integration of machine learning datasets through Pelican origins for improved data accessibility and processing.
- Develope and teste models from small prototypes to extensive systems within an HTCondor-managed workflow, significantly optimizing computational throughput.
- Create comprehensive tutorials and practical guides to facilitate the adoption of PyTorch and Pelican integration across the machine learning community.

Pelican Website: https://pelicanplatform.org/

Related tools:

- HTCondor: https://htcondor.org/

## Dataset using:

[Fashion-mnist](https://www.kaggle.com/datasets/zalando-research/fashionmnist)

[ImageNet](https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/description)

[ImageNet Mini](https://github.com/fastai/imagenette)

## Current data under `chtc/PUBLIC/hzhao292` name space

| Size | File Name                                  |
| ---- | ------------------------------------------ |
| 22M  | fashion-mnist_test.csv                     |
| 5.4M | fashion-mnist_test.zip                     |
| 127M | fashion-mnist_train.csv                    |
| 33M  | fashion-mnist_train.zip                    |
| 161G | ILSVRC                                     |
| 27M  | ImageNet                                   |
| 156G | imagenet-object-localization-challenge.zip |
| 1.5G | imagenette2                                |
| 1.5G | imagenette2.tgz                            |

### Data for Benchmarking1

| Size | File Name               |
| ---- | ----------------------- |
| 22M  | fashion-mnist_test.csv  |
| 5.4M | fashion-mnist_test.zip  |
| 127M | fashion-mnist_train.csv |
| 33M  | fashion-mnist_train.zip |

### Data for Benchmarking2

| Size | File Name                                  |
| ---- | ------------------------------------------ |
| 161G | ILSVRC                                     |
| 156G | imagenet-object-localization-challenge.zip |
| 1.5G | ImageNetMini                               |
| 1.5G | ImageNetMini.tgz                           |

For ImageNet standard data, train file is under ILSVRC/Data/CLS-LOC/train. 

Under this path, there is 1000 directories named with corresponding classes of these images inside the directories. 

Val and Test follow the same rool after preparation. 

ImageNetMini is a subset for conveniently testing.  It only have 10 classes. About 1000 images in each class.



## Newest Update

