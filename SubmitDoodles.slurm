#!/bin/bash
#
#SBATCH --job-name=MLfinal
#SBATCH --output=qd_out.txt
#SBATCH --error=qd_error.txt
#
#SBATCH --partition=research
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=20GB
#SBATCH --gres=gpu:titanxp:1

module load cuda

export CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=2 /home/epeake/venv/mainenv/bin/python3 -u QuickDrawModel.py -W
