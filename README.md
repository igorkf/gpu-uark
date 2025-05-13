# gpu-uark

## Logging on the cluster

There are many ways to interact with the cluster.

### From the portal   
Access https://hpc-portal2.hpc.uark.edu/pun/sys/dashboard and put credentials. For example, my username is `igorf` (without the @uark.edu)

### From the terminal
#### Windows users
1. Open Windows PowerShell
2. Run:
```
ssh USERNAME@hpc-portal2.hpc.uark.edu
```
where `USERNAME` is your username.

#### Mac and Linux users
1. Open the Terminal
2. Run:
```
ssh USERNAME@hpc-portal2.hpc.uark.edu
```
where `USERNAME` is your username.

Note: for both cases, if you are outside campus, you need to run first `ssh USERNAME@hpc-portal2.hpc.uark.edu -p 2022` to authenticate, then run `ssh USERNAME@hpc-portal2.hpc.uark.edu`.

## Copying files to the cluster

### From the portal
1. Download the repository code: https://github.com/igorkf/gpu-uark/archive/refs/heads/main.zip
2. Access the portal
3. Go to Files > Home Directory
4. Click Upload
5. Upload the file `gpu-uark-main.zip`

### From the terminal
1. Open a **new** terminal (PowerShell for Windows or Terminal for Mac or Linux). Do not login to the cluster yet.
2. Go to the folder where you downloaded the zip file (e.g., `Downloads`):
    ```
    cd Downloads
    ```
3. Copy the repository code to the cluster (change `USERNAME` by your username):
    ```
    scp gpu-uark-main.zip USERNAME@hpc-portal2.hpc.uark.edu:/home/USERNAME
    ```

## Unzipping folders
1. Login on the cluster:
    ```
    ssh USERNAME@hpc-portal2.hpc.uark.edu
    ```

2. Unzip the main file and then enter the folder:
    ```
    unzip gpu-uark-main.zip
    cd gpu-uark-main
    ```

5. Unzip the data:
    ```
    unzip data.zip
    ```

## Running jobs on the cluster
Now, we are ready to run the tasks we need.

### Preprocessing the data
1. To preprocess the data, run:
    ```
    sbatch 1-preprocess.sh
    ```
2. To check job status, run:
    ```
    squeue -u USERNAME
    ```

Some files will be written to `output`.   
Check the genotypic data:
```
head output/geno_ok.csv
```
Check the phenotypic data:
```
head output/train_val_test.csv
```

Bonus (optional):   
Dr. Fernandes has two private nodes that have more computational power. To preprocess the data using the private nodes, we have to access a different operation system (OS) and then run the task:

1. Connect to the other OS:
    ```
    ssh pinnacle-l12
    ```

2. Change to the code's directory:
    ```
    cd gpu-uark-main
    ```

3. Run the job:
    ```
    sbatch 1-preprocess-example-condo.sh
    ```

The output should be the same.