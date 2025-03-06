#!/bin/bash
#SBATCH -t 02:00:00           # Time limit: 4 hours
#SBATCH --nodes=1             # Number of nodes
#SBATCH --ntasks=32           # Number of tasks
#SBATCH --gpus-per-node=1     # GPUs per node
#SBATCH -A standby            # Account/Partition: standby
#SBATCH --output=job_output7.log  # Standard output log
#SBATCH --error=job_error7.log    # Error log

module load conda/2024.09
conda activate CS587

# Change to the directory where driver.py is located
/home/rnahar/Desktop/hw2_skeleton/hw2_imageclassifier_skeleton

# Run the script
# python main.py --cnn --kernel 3 --stride 3 --batch-size 100 --device cuda

# Question 2
# python main.py --batch-size 100 --cnn --lr 1e-3 --kernel 3 --stride 3 --device cuda --num-epochs 100
# python main.py --batch-size 100 --cnn --lr 1e-3  --kernel 14 --stride 1 --device cuda --num-epochs 100

# quesion3 
# python main.py --batch-size 100 --cnn --lr 1e-3 --kernel 5 --stride 1 --device cuda --num-epochs 100

# question 4
# python main.py --shuffle-label --cnn --lr 1e-2 --kernel 5 --stride 1 --batch-size 100
# python main.py --batch-size 100 --cnn --lr 1e-2 --shuffle-label --device cuda --num-epochs 100

# question 5
# python main.py

# question 6
# python main.py --cnn --rot-flip --device cuda

# question 7
python main.py --cgcnn --rot-flip --device cuda --lr 1e-3