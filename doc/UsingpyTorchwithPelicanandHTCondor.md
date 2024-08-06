# Using pyTorch with Pelican, and HTCondor



This is an integrated tutorial for using pytorch with Pelican and HTCondor. Before proceeding, ensure you have a CHTC account. If you don't have one, see [How to Request a CHTC Account](https://chtc.cs.wisc.edu/uw-research-computing/account-details.html).

For beginners, each step here can be a little bit challenged. We will disucss typical case for each step here to provide an overview and help you get started quickly.  But for each step, there is a detailed documentation for further exploration. For you reference, all the related tutorial are listed here.

**Related Tutorial:**

[Create a Portable Python Installation with Miniconda](https://chtc.cs.wisc.edu/uw-research-computing/conda-installation.html).

[Handling Data in Jobs](https://chtc.cs.wisc.edu/uw-research-computing/file-availability)

[Practice: Submit HTC Jobs using HTCondor](https://chtc.cs.wisc.edu/uw-research-computing/htcondor-job-submission)

[Run Machine Learning Jobs](https://chtc.cs.wisc.edu/uw-research-computing/machine-learning-htc)

[Use GPUs](https://chtc.cs.wisc.edu/uw-research-computing/gpu-jobs.html)

[Submit High Memory Jobs](https://chtc.cs.wisc.edu/uw-research-computing/high-memory-jobs)

## 1. Create Your Environment for Your Job

HTC's executed point usually only have some necessary softwares installed. Therefore, if you're using PyTorch and other libraries, you need to create an environment for your job.

Following this tutorial: [Create a Portable Python Installation with Miniconda](https://chtc.cs.wisc.edu/uw-research-computing/conda-installation.html). You should have a tar file `env-name.tar.gz`. In most cases, the file is larger than 1GB, you should use the large data filesystem staging provided by CHTC.

To check the size of a file, do:

```shell
du -sh filename
```

## 2. Managing your needed files

Here's an overview of how to handle files of different sizes. Since files larger than 100MB, or even 1GB, are common in machine learning training jobs, we will focus on using `/staging` here. Beyond these, OSDF protocol links can also be passed directly to the HTCondor execute point!

| Input Sizes                                                  | Output Sizes             | Link to Guide                                                | File Location | How to Transfer                                      | 
| :----------------------------------------------------------- | :----------------------- | :----------------------------------------------------------- | :------------ | :--------------------------------------------------- |
| 0 - 100 MB per file, up to 500 MB per job                    | 0 - 5 GB per job         | [Small Input/Output File Transfer via **HTCondor**](https://chtc.cs.wisc.edu/uw-research-computing/file-availability.html) | `/home`       | **submit file**; filename in `transfer_input_files`  |
| 100 MB - 1 GB per repeatedly-used file                       | Not available for output | [Large Input File Availability Via **Squid**](https://chtc.cs.wisc.edu/uw-research-computing/file-avail-squid.html) | `/squid`      | **submit file**; http link in `transfer_input_files` |  
| 100 MB - TBs per job-specific file; repeatedly-used files > 1GB | 4 GB - TBs per job       | [Large Input and Output File Availability Via **Staging**](https://chtc.cs.wisc.edu/uw-research-computing/file-avail-largedata.html) | `/staging`    | **job executable**; copy or move within the job      |    



### 2.1 Via `transfer_input_files`

- small file: on your ap
- big file: pelican protocol

### 2.2 Via `/staging`

`/staging` is a distinct location for staging data that is too large to be handled at scale via the default HTCondor file transfer mechanism.This location should be used for jobs that require input files larger than 100MB and/or that generate output files larger than 3-4GB.

#### 2.2.1 Check your Quotas

For individual directories, your quotas are printed on login. For group directories at HTC `/staging`, HPC `/home`, HPC `/scratch`, you can retrieve your quotas using the command

```shell
get_quotas /path/to/group/directory
```

#### 2.2.2 Request Quota Changes

If you want to start using `/staging`, request a Quota change via this [form](https://uwmadison.co1.qualtrics.com/jfe/form/SV_0JMj2a83dHcwX5k). If you do not receive an automated email from [chtc@cs.wisc.edu](mailto:chtc@cs.wisc.edu) within a few hours of completing the form, OR if you do not receive a response from a human within two business days (M-F), please email [chtc@cs.wisc.edu](mailto:chtc@cs.wisc.edu).

#### 2.2.3 Moving you file to `/staging`

Once assigned, you should have some space under `/staging/your-username`. To transfer the large file, such as our our `env-name.tar.gz`, if the file in on your local linux/mac computer, use `scp` command via CHTC's **transfer sever** instead of a CHTC submit server. 

```shell
$ scp /localpath/env-name.tar.gz username@transfer.chtc.wisc.edu:/staging/username/ 
```

If the file is one your access point(submit sever), then you can use `cp` to move the file directly.

```shell
$ cp env-name.tar.gz /staging/username/ 
```

#### 2.2.4 Using Files in /staging in a Job

After submitting your job to HTC's execute point, everything will happen under the working directory on the execute point. Therefore, to use the large file, first move the file from `/staging` to the working directory of the job. You can achieve this by adding the `cp` command to your job executable. Make sure to remove the file you copied after your job completed.



For example, a `.sh` file as executable:

```shell
#!/bin/bash
#
# First, copy the compressed tar file from /staging into the working directory,
# and un-tar it to reveal your large input file(s) or directories:
cp /staging/username/large_input.tar.gz ./
tar -xzvf large_input.tar.gz
#
# Command for myprogram, which will use files from the working directory
./myprogram large_input.txt myoutput.txt
#
# Before the script exits, make sure to remove the file(s) from the working directory
rm large_input.tar.gz large_input.txt
#
# END
```



Here is an example using the environment package we discussed before. 

```shell
#!/bin/bash

# have job exit if any command returns with non-zero exit status (aka failure)
set -e

# replace env-name on the right hand side of this line with the name of your conda environment
ENVNAME=env-name

# if you need the environment directory to be named something other than the environment name, change this line
export ENVDIR=$ENVNAME

# these lines handle setting up the environment; you shouldn't have to modify them
export PATH
mkdir $ENVDIR

# Copying the large file (enviornment package here) from /staging/username
cp /staging/username/env-name.tar.gz ./

tar -xzf $ENVNAME.tar.gz -C $ENVDIR
. $ENVDIR/bin/activate

# modify this line to run your desired Python script and any other work you need to do
python3 bm3.py -e 3 -a vgg16 -b 32 -j 4

# Before the script exits, make sure to remove the file(s) from the working directory
rm $ENVNAME.tar.gz
```

## 3. Submit job

In this section we will talk about submitting a job via HTCondor.

You would need a `.sub` file specify your need for the job. Take our benchmark python script `bm.py` as an example, we will have a file named `bm.sub`:

```shell
executable              = bm.sh

log                     = bm.log
output                  = bm.out
error                   = bm.err

transfer_input_files    = bm.py ,pelican://osg-htc.org/chtc/PUBLIC/hzhao292/ImageNetMini.tgz
should_transfer_files   = Yes
when_to_transfer_output = ON_EXIT

request_cpus            = 4
request_memory          = 16G
request_disk            = 10G
request_gpus            = 1

Requirements = (Target.HasCHTCStaging == true)

+WantGPULab = true
+GPUJobLength = "short"

queue 1
```

Here, we specify the name our executable file, and also names of log file, file storaged standard output and file for error message.

Then are files we will need during the job running, this will included the `bm.py` script it self, and the data it used. For the env package, we are using it through staging, therefore, it shouldn't show up here. Instead, only add `Requirements = (Target.HasCHTCStaging == true)` to let HTCondor know you have file in staging.

Then in the file, tells HTCondor what kinds of resourse you job need, like `request_memory` and `request_cpus` for how much memory and how many cpu cores are needed. `request_disk` for the disk size the job will need.

If you want to utilize GPUs, add `request_gpus`.

Finally, the queue statement tells HTCondor that you are done describing the job, and to send it to the queue for processing.

Now, submit your job to HTCondor’s queue using `condor_submit`:

```shell
condor_submit bm.sub
```

The `condor_submit` command actually submits your jobs to HTCondor. If all goes well, you will see output from the `condor_submit` command that appears as:

```
Submitting job(s)...
3 job(s) submitted to cluster 36062145.
```

To check on the status of your jobs in the queue, run the following command:

```
$ condor_q
```

The output of `condor_q` should look like this:

```
-- Schedd: ap2002.chtc.wisc.edu : <128.105.68.113:9618?... @ 08/06/24 15:39:29
OWNER     BATCH_NAME     SUBMITTED   DONE   RUN    IDLE  TOTAL JOB_IDS
username  ID: 3606214    8/5 12:31     _    1       _      1 36062145.0-2
   
1 jobs; 0 completed, 0 removed, 0 idle, 1 running, 0 held, 0 suspended
```

Usually, your job will need to wait for a little time to run. Waiting time will depend on the avaliablity and your request. Becasue of higher request may result in long waiting time, we will recommend you to test all things work well by submitting a sample job without requesting too many resource. You can run the `condor_q` command periodically to see the progress of your jobs.