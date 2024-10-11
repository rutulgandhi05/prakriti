#!/bin/bash

#SBATCH -J thechosenone
#SBATCH -o output.txt
#SBATCH -e error.txt
#SBATCH -c 8
#SBATCH -n 1
#SBATCH -t 10:00:00 
#SBATCH -p gpu_8
#SBATCH --gres=gpu:4
#SBATCH --mem=64000
#SBATCH --mail-type ALL
#SBATCH --mail-user rutul.gandhi@uni-ulm.de

source venv/bin/activate

pip install -r thechosenone/requirements.txt
python thechosenone/main.py


