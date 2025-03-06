#!/bin/bash
#SBATCH -t 00:30:00           # Time limit: 4 hours
#SBATCH --nodes=1             # Number of nodes
#SBATCH --ntasks=32           # Number of tasks
#SBATCH --gpus-per-node=1     # GPUs per node
#SBATCH -A standby            # Account/Partition: standby
#SBATCH --output=job_output.log  # Standard output log
#SBATCH --error=job_error.log    # Error log

module load conda/2024.09
conda activate CS587

# Change to the directory where driver.py is located
/home/rnahar/Desktop/hw2_skeleton/hw2_hmcmlp_skeleton

# Run the script
python main.py Desktop/hw2_skeleton/data/MNIST/raw --depth shallow

# deep
# python main.py Desktop/hw2_skeleton/data/MNIST/raw --depth shallow
