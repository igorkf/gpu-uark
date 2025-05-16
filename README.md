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

Some files will be written to `logs` and `output`.  
Check the log:
```
cat logs/prep_geno.log
```

Check the genotypic data:
```
head output/geno_ok.csv
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

### Configuring Python environment with GPU dependencies
1. Run the configuration task:
    ```
    sbatch 2-config_python_env.sh
    ```

Check the logs:
```
cat logs/config_python_env.out
```

See that a new Python environment was created. It is packed in a single folder called `myenv`:
```
ls -lht myenv
```
It has all the libraries (dependencies) we need to train the models using Python.

### Performing feature engineering and other steps
Now that we have the genotypic data cleaned (`output/geno_ok.csv`), we can perform other steps such as feature engineering to generate the final data for training and evaluating the models.

1. Run:
    ```
    sbatch 3-create-datasets.sh
    ```

Check the output files:
```
ls -lht output
```

In this case, the training data comprises trials from 2021, whereas validation data trials from 2022. Thus, this is a case of CV0-Year, where environments from a future year are untested. You could test other cross-validation scenarios, but this would demand some changes in the code.

### Train a LightGBM model
With all the data available, now we fit and evaluate the models. 

1. Fit and evaluate the model:
    ```
    sbatch 4-train_lgbm.sh 
    ```

Check the logs:
```
cat logs/train_lgbm.logs
```

Check mean predictive ability across environments:
```
cat logs/train_lgbm.log | grep "mean env corr:"
```

Predictions are stored at `output/pred_lgbm.csv`.

### Train a neural network with PyTorch
As the cluster has GPU available, we can fit neural networks using the GPU.   

1. Run:
    ```
    # for CPU
    sbatch 5-train_nn_cpu.sh
    ```
    ```
    # for GPU
    sbatch 5-train_nn_gpu.sh
    ```

It will take some minutes. We can check the job status. For example:
```
squeue -u igorf
```
gives:
```
(myenv) c1601:igorf:~/repos/gpu-uark$ squeue -u igorf
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
            716165   cloud72   nn_cpu    igorf  R       7:07      1 c1601
            716169     gpu06   nn_gpu    igorf  R       4:59      1 c1702
```

One model is using CPUs (`partition=cloud72`) and the other is using GPU (`partition=gpu06`).   


#### Checking GPU metrics
1. Open a new tab of your terminal and connect to the node displayed above (in this exmaple, it was ran at `c1702`):
    ```
    ssh c1702
    ```

2. Show GPU metrics every 0.1 seconds:
    ```
    watch -n 0.1 nvidia-smi
    ```

See the usage (%), temperature (C), etc.   
In this case, as the data is not that big, the usage is about 20% only.   
The results for both cases (CPU) and (GPU) should be similar, although using the GPU can be non-deterministic for some operations.

3. Logout from the node:
    ```
    logout
    ```
   
Predictions are stored at `output/pred_nn_*.csv`.

## Coding in the cluster using an Integrated Development Environment (IDE)

Two IDEs are pretty good for coding in Python:

- [VSCode](https://code.visualstudio.com/Download)
- [Cursor](https://www.cursor.com/downloads)

Cursor is a new IDE with built-in LMMs. You can code and use ChatGPT-like tools inside Cursor to help you code better. You can even use your OpenAI API and choose different LLM models (gpt, claude, gemini, etc.).

### Connecting to the cluster via SSH

After downloading and installing the IDE, we need to set some configurations.

1. Hit `Ctrl + Shift + P`, type `Connect to Host...` and hit `Enter`
2. Choose `Configure SSH Hosts...`
3. Choose the Users option (e.g., for my Windows it it `C:\Users\igork\.ssh\config`
4. Type the following (changing USERNAME by your username):
    ```
    Host c1602
      HostName c1602
      ProxyJump hpc-portal2.hpc.uark.edu
      User USERNAME

    Host hpc-portal2.hpc.uark.edu
      HostName hpc-portal2.hpc.uark.edu
      User USERNAME
    ```

The configuration above connects to the login node and "jumps" to the given node (in this case, `c1602`). This means the IDE will connect to the node `c1602`, which uses the `cloud72` partition. If you want to connect to another node, change the `Host NODE` and `HostName NODE` parts.   

5. Hit again `Ctrl + Shift + P`, type `Connect to Host...` and hit `Enter`
6. Choose the node you typed above.

A new window will pop up, and, if the connection was successful, you now are connect to the cluster. Open a terminal inside your IDE and you should see something like this:
```
c1602:igorf:~$
```

This means we are using the node `c1602`. For now on, you can code and run computational things in the terminal if you need.

### Potential problems

#### Connection unauthorized
We tried to connect to a specific node, but this node doesn't "know" who is trying to connect. In this case, we need to add our public key in the node's authorized keys list. 

1. Open a new terminal on your local machine (note connected to the cluster)
2. Create a public key:
    ```
    ssh-keygen
    ```
3. Follow the steps shown on the screen

4. Print your public key and copy it (change FILENAME to your filename)
    ```
    # in Windows
    type C:\Users\igork\.ssh\FILENAME.pub
    ```
    ```
    # mac and Linux
    cat ~/.ssh/FILENAME.pub
    ```
5. Login on the cluster (change USERNAME to your username):
    ```
    ssh USERNAME@hpc-portal2.hpc.uark.edu
    ```

6. Connect to the node chosen above (in this case, `c1602`):
    ```
    ssh c1602
    ```

7. Open the authorized keys file:
    ```
    nano cat ~/.ssh/authorized_keys
    ```

6. Go to the last line and paste the content you copied in step 2

7. Hit `Ctrl + X`, then `Y`, then `Enter`

### Running R
1. Load the R modules:
    ```
    module load intel/21.2.0 mkl/21.3.0 R/4.3.0 gcc/11.2.1
    ```

2. Run:
    ```
    R
    ```
