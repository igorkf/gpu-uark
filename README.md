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
5. Upload the folder zip file `gpu-uark-main.zip`

### From the terminal
1. Open a new terminal (PowerShell for Windows, or Terminal for Mac or Linux).
2. Go to the folder where you downloaded the zip file (e.g., `Downloads`):
    ```
    cd Downloads
    ```
3. Copy the repository code to the cluster (change `USERNAME` by your username):
    ```
    scp gpu-uark-main.zip USERNAME@hpc-portal2.hpc.uark.edu:/home/USERNAME
    ```


