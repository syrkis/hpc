# High Performance Cluster (HPC) Demo

This repo demos how to train a PyTorch based neural model on ITU's HPC.
Specifically we train a variational auto encoder to generate digits.


## Setup
1. Connect to the ITU network (VPN or WiFi).
2. Log on to the hpc: `ssh username@hpc.itu.dk` (use your ITU username).
3. Clone this repository into your home folder in the server: `git clone git@github.com:syrkis/hpc.git`.
4. Change directory into the repo: `cd ./hpc/`.
5. run `module --ignore-cache load Python/3.7.4-GCCcore-8.3.0`.
5. Create a virutal environment to hold your python dependencies: `virtualenv venv`
6. Activate the environment: `source venv/bin/activate`.
7. Install the requirements: `pip install -r requirements.txt`

## Usage
Now you have a remote copy of your repo on the server, and we've installed all the dependencies you need.
To run a job type `sbatch jobs/gpu.job`.
The top of the file specifies the clutser settings with which the job should be run.
Next we load some libraries we need, beforre activating the python venv, and then calling the python script.
Remember, the job does *not* run on the machine that you've logged in to.
