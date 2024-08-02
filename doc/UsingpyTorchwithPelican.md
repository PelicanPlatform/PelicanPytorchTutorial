

# Using pyTorch with Pelican

[toc]



## Step1: Have your Pelican and PyToch installed. 

For detailed of Pelican Installation, see [here](https://docs.pelicanplatform.org/install)

For macOS and Linux, run `which pelican` to verify your installation. And use following command to start your first pelican operation!

```shell
$ pelican object get pelican://osg-htc.org/chtc/PUBLIC/hzhao292/test.txt test.txt
$ ls
test.txt
$ cat test.txt
Hello, World!
```

For pytorch installation, see [here](https://pytorch.org/get-started/locally/).

## Access Pelican's data

### Option 1: Using command line 

Before grabing your data, I believe you have known the location of it. In this case, you should have known your target origin's federation, name space, and path to target file. 

Then you can use `object get` command to download your data to local destination. For more details, read [this](https://docs.pelicanplatform.org/getting-data-with-pelican/client)

```{shell}
pelican object get pelican://<federation-url></namespace-prefix></path/to/file> <local/path/to/file>
```

An example for you to try out:

``````
pelican object get pelican://osg-htc.org/chtc/PUBLIC/hzhao292/ImageNetMini.tgz ImageNetMini.tgz
``````

You should see a progress bar output and eventually a file named `ImageNetMini.tgz` within your local directory:

```shell
$ pelican object get pelican://osg-htc.org/chtc/PUBLIC/hzhao292/ImageNetMini.tgz ImageNetMini.tgz 
ImageNetMini.tgz 1.34 GiB / 1.45 GiB [========================>--] 0s ] 0.00 b/s

```

If the target object is a directory, you can download the whole directory with anything in it using **--recursive** flag.

 ```pelican object get pelican://osg-htc.org/chtc/PUBLIC/hzhao292/ImageNetMini --recursive ImageNetMini```

> Note: We do not recommend this way, because it will transfer file by file, which will result in many requests. And it's slow.

### Option 2: Using pelicanfs

Using the command line is easy, buy you may want to do some more complicated operation, or just want to integrate it to your python code to make them streamlined and consist. In this way, we have `pelicanfs` library, which implement the [`fsspec`](https://filesystem-spec.readthedocs.io/en/latest/index.html). `fsspec` exists to provide a familiar API that will work the same whatever the storage backend. This means, if you have some data on google cloud or s3, you can access them all the same! (For more detail information,  please read [fsspec's document](https://filesystem-spec.readthedocs.io/en/latest/index.html))

To install pelicanfs, run:

```shell
pip install pelicanfs
```

Frist: Use fsspec with protocal

In this way, we initialize an OSDF File System, then we can access our file in it. 

```python
import fsspec

fs = fsspec.filesystem("osdf") 
valfile_path = "/chtc/PUBLIC/hzhao292/ImageNetMini/val"
fs.ls(valfile_path)
```

Second: Using Pelican File System

If you are using the pelican file system, you need to pass the discovery URL of the Federation because it doesn't know which federation are you come from, in this case, We are passing OSDF's discovery URL.

```python
from pelicanfs.core import PelicanFileSystem

fs = PelicanFileSystem("pelican://osg-htc.org")
valfile_path = "/chtc/PUBLIC/hzhao292/ImageNetMini/val"
fs.ls(valfile_path)
```

In a nutshell, you should either pass the federation protocal to fsspec's `filesystem`, or discovery URL of your federation to `PelicanFileSystem` in pelicanfs. 

To download file to your local file system, use `get()`.

```python
fs.get("/chtc/PUBLIC/hzhao292/ImageNetMini.zip","./")
```



