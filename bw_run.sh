#!/bin/bash

#SBATCH -J thechosenone
#SBATCH -o output.txt
#SBATCH -e error.txt
#SBATCH -c 8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 10:00:00 
#SBATCH -p gpu_4_a100
#SBATCH --gres=gpu:2
#SBATCH --mem=64000
#SBATCH --mail-type ALL
#SBATCH --mail-user rutul.gandhi@uni-ulm.de

source $HOME/venv11/bin/activate

python thechosenone/main.py


