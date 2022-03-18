# HPC

This is a template for running PyTorch projects on a Slurm cluster.
The hpc.job file can be called from the access node using ```sbatch hpc.job```.
An enviornment folder is required, and it can be created by running:
```virtualend -p $(which python3) env && source env/bin/activate && pip install -r requirements.txt && deactivate```.
If you want to store your data on the HPC
(as opposed to downloading it on the fly through the ```Dataset``` class)
make a ```data``` folder and upload the files from your computer using ```scp```.
That's about it.


