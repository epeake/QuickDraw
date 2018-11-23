#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --output=test_out.txt
#SBATCH --error=test_error.txt
#
#SBATCH --partition=research
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=1GB
#SBATCH --gres=gpu:titanxp:1

module load cuda

CUDA_VISIBLE_DEVICES=2 /home/epeake/venv/mainenv/bin/python3 -u QuickDrawModel.py -W
