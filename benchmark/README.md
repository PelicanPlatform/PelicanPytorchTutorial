Benchmark1 is a notebook using fashion-MNIST dataset. Benchmark2 using ImageNet dataset.

`bm3.py` is a python script version of benchmark2, which will be convenient to run and set different arguments in terminal.

`remote_image_folder.py` is the code of  RemoteImageFolder we use in Benchmark2. We set it apart because it will be difficult for mutiprocessing using jupyter notebook.



For `bm3.py`

## Flag:

-a: choose architecture from resnet50 and vgg16, default is resnet50

-e: set the number of epochs you want to run.

-b: set the mini-batch size of training, default is 256

-j: Set the number of data loading workers, default is 2.

## Example:

Running the benchmark script using model vgg16 for 10 epochs. Using batch size 32 and one dataloader worker.

```{shell}
python bm3.py -a vgg16 -e 10 -b 32 -j 1
```

