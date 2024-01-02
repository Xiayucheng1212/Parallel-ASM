#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

make asm_gpu
make asm_blocked

srun ./asm_gpu ./testcase/MAX_T ./testcase/MAX_P gpu_out
srun ./asm_blocked ./testcase/MAX_T ./testcase/MAX_P blocked_out