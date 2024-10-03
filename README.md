# VDT
Laboratory work on the article “VDT: universal video diffusion transformers using mask modeling”, authors Mikhelson German, Enrico Pallota, Prof. Dr. Jurgen Gall

<img src="VDT.png" width="700">

## Introduction
This report outlines the tasks performed in implementing and training the Video Diffusion Transformer (VDT) model for video prediction and unconditional generation tasks using the CityScapes dataset. The primary objectives included implementing data preprocessing, metric evaluation, training/validation functions, and reproducing experimental results. The results, challenges, and suggestions for future improvements are discussed.

## Getting Started

- Python3, PyTorch>=1.8.0, torchvision>=0.7.0 are required for the current codebase.
- To install the other dependencies, run
    ```
    conda env create -f environment.yml
    conda activate VDt_new
    ```
- For running the code you can use one of the bash scripts. Below are examples for running the code in both train and test modes on single or multiple GPUs.
    - **Train** (for Test you only need to change `--run_mode` to "test"):
        - **Single GPU**:
        ```
        python3 main.py --model VDT-L/2 --vae mse --image-size 128 \
        --f None --num-classes 1 --batch_size 6 --cfg-scale 4 --num-sampling-steps 500 --seed 0 \
        --num_frames 16 --epoch 100 --ckpt vdt_model_500_1061.pt --mode paral --run_mode train
        ```
        - **Multiple GPUs**:
        ```
        torchrun --nproc_per_node=4 main.py --model VDT-L/2 --vae mse --image-size 128 \
        --f None --num-classes 1 --batch_size 6 --cfg-scale 4 --num-sampling-steps 500 --seed 0 \
        --num_frames 16 --epoch 100 --ckpt vdt_model_500_1061.pt --mode paral --run_mode train
        ```


## Checkpoint
We now provide checkpoint for CityScapes Video Prediction and Unconditional Generation. To understand which model you need, be sure to read the report, which describes all the features (the report is attached along with the models). You can download it from <a href="https://drive.google.com/drive/folders/14J5NEeaDxDMB9R_0RDkqkvxwPEzKwy_g?usp=sharing">here</a>.


## Orinal code
This is the original repository <a href="https://github.com/RERV/VDT?tab=readme-ov-file">here</a>
