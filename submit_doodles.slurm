#!/bin/bash
#
#SBATCH --job-name=MLfinal
#SBATCH --output=qd_out.txt
#SBATCH --error=qd_error.txt
#
#SBATCH --partition=research
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=1GB
#SBATCH --gres=gpu:titanxp:1

chmod +x merge_doodles.sh
./merge_doodles.sh

module load cuda

CUDA_VISIBLE_DEVICES=0 /home/epeake/venv/mainenv/bin/python3 -u QuickDrawModel.py -W
