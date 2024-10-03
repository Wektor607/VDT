#!/bin/bash
#SBATCH --partition=A40devel
#SBATCH --time=1:00:00
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --ntasks=4

#SBATCH --output=/home/s17gmikh/VDT/log_outputs/log/VDT_Benchmark_%j.output
#SBATCH --error=/home/s17gmikh/VDT/log_outputs/error/VDT_Benchmark_%j.error

#SBATCH --mail-type=ALL
#SBATCH --mail-user=s17gmikh@uni-bonn.de

mkdir -p /home/s17gmikh/VDT/log_outputs
mkdir -p /home/s17gmikh/VDT/log_outputs/log
mkdir -p /home/s17gmikh/VDT/log_outputs/error

source /home/s17gmikh/miniconda3/etc/profile.d/conda.sh

conda activate VDt_new

cd /home/s17gmikh/VDT

module load Python
module load CUDA/11.7.0
module purge

export OMP_NUM_THREADS=4

torchrun --master_port=6000 --nproc_per_node=4 main.py --model VDT-L/2 --vae mse --image-size 128 \
        --f None --num-classes 1 --batch_size 16 --cfg-scale 4 --num-sampling-steps 500 --seed 0 \
        --num_frames 30 --epoch 100 --ckpt vdt_model_500_979.pt --mode paral --run_mode test --task_mode video_pred