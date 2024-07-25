

# Using pyTorch with Pelican



## Step1: Have your Pelican and PyToch installed. 

For detailed of Pelican Installation, see [here](https://docs.pelicanplatform.org/install)

For macOS and Linux, run `which pelican` to verify your installation. And use `pelican object copy pelican://osg-htc.org/ospool/uc-shared/public/OSG-Staff/validation/test.txt .` to start your first pelican operation!

## Access Pelican's data

### Option 1: Using command line 

Before grabing your data, I believe you have known the location of it. In this case, you should have known your target origin's federation, name space, and path to target file. 

Then you can use `object get` command to download your data to local destination. For more details, read [this](https://docs.pelicanplatform.org/getting-data-with-pelican/client)

```{shell}
pelican object get pelican://<federation-url></namespace-prefix></path/to/file> <local/path/to/file>
```

An example for you to try out:

```pelican object get pelican://osg-htc.org/chtc/PUBLIC/hzhao292/ImageNetMini.tgz ImageNetMini.tgz```

You should see a progress bar output and eventually a file named `downloaded-test.txt` within your local directory:

```
$ pelican object get pelican://osg-htc.org/ospool/uc-shared/public/OSG-Staff/validation/test.txt downloaded-test.txtdownloaded-test.txt 
14.00 b / 14.00 b [==============================================================================] Done!
$ lsdownloaded-test.txt$ cat downloaded-test.txtHello, World!
```

```python
import pelicanfs
import torch

from pelicanfs.core import PelicanFileSystem

fs = PelicanFileSystem()
valfile_path = "/chtc/PUBLIC/hzhao292/ImageNetMini/val"
fs.ls(valfile_path)
```



If the target object is a directory, you can download the whole directory with anything in it using **--recursive** flag.

 ```pelican object get pelican://osg-htc.org/chtc/PUBLIC/hzhao292/ImageNetMini --recursive  ImageNetMini```

### Option 2: Using pelicanfs

```{shell}
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

If you are using the pelican file system, you need to pass the discovery URL of the Federation, in this case. We are passing OSDF's discovery URL.

```python
from pelicanfs.core import PelicanFileSystem

fs = PelicanFileSystem("pelican://osg-htc.org")
valfile_path = "/chtc/PUBLIC/hzhao292/ImageNetMini/val"
fs.ls(valfile_path)
```

