#!/bin/sh
#SBATCH --cpus-per-task=2
#SBATCH --mem=8gb
#SBATCH --partition=gpu
#SBATCH --gpus=2

#SBATCH --time=24:00:00

#SBATCH --job-name=part3_interpolation
#SBATCH --mail-type=all
#SBATCH --mail-user=andrew.watson@ufl.edu
#SBATCH --output=serial_%j.log

pwd; hostname; date
module purge
#echo "Loading python"
#module load python
#echo "Loading cuda"
#module load cuda
echo "Loading tensorflow"
module load tensorflow
echo "Running python script"
python /home/andrew.watson/Deep-Learning-and-CG-Project---Frame-Interpolation/train.py
date
