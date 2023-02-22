# High Performance Cluster (HPC) Demo

This repo demos how to train (on GPU) and use (in parallel) a PyTorch based neural model on ITU's HPC.
Specifically we train a variational auto encoder to generate digits.


## Setup
1. Connect to the ITU network (VPN or WiFi).
2. Log on to the hpc: `ssh [user]@hpc.itu.dk` (use your ITU username) and switch to zsh by running `zsh`.
3. Make the folders that are expected to be there by my code: `mkdir models plots logs`.
4. Clone this repository into your home folder in the server: `git clone git@github.com:syrkis/hpc.git hpc`.
5. Change directory into the repo: `cd ./hpc/`.
6. Install the python dependencies we use in this example: `sbatch jobs/setup.job` (this will take a while as it installs python deps.).
7. You can monitor progress using `logs/[job_id].log`.


## Usage
Documentation for the cluster is avaiable at hpc.itu.dk (when you are connected to the network).
In actually run a job on the cluster, you must call a job file with `sbatch` (as we did with setup).
The job files initially specify the cluster configurations, after which the commands below are run consecutively.
In the case of `jobs/gpu.job` we first declare that we want to use a gpu (among other specs).
Then we load in our version of Python and CUDA, activate out environment and call `main.py`.

To see what jobs are currently runnning, run `sbatch`. To cancel a job run `scancel [jobid]`.
That pretty much it. You can also 