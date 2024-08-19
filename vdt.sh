#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1

#SBATCH --output=/home/s17gmikh/VDT/log_outputs/log/VDT_Benchmark_%j.output
#SBATCH --error=/home/s17gmikh/VDT/log_outputs/error/VDT_Benchmark_%j.error

#SBATCH --mail-type=ALL
#SBATCH --mail-user=s17gmikh@uni-bonn.de

mkdir -p /home/s17gmikh/VDT/log_outputs
mkdir -p /home/s17gmikh/VDT/log_outputs/log
mkdir -p /home/s17gmikh/VDT/log_outputs/error

source /home/s17gmikh/miniconda3/etc/profile.d/conda.sh

conda activate VDT

module load Python
module purge

cd /home/s17gmikh/VDT

python3 train.py --model VDT-S/2 --vae mse --image-size 128 --f None --num-classes 1 --batch_size 8\
                --cfg-scale 4 --num-sampling-steps 16 --seed 0 --num_frames 30 --device cuda

                
