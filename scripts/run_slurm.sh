#!/bin/bash
#SBATCH --job-name=rand-smooth
#SBATCH --account=fc_wagner
#SBATCH --partition=savio3_gpu
# Number of nodes:
#SBATCH --nodes=1
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=2
# Processors per task (please always specify the total number of processors twice the number of GPUs):
#SBATCH --cpus-per-task=2
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included, GTX2080TI
#SBATCH --gres=gpu:GTX2080TI:2
#xSBATCH --qos=v100_gpu3_normal
#SBATCH --time=24:00:00
#xSBATCH --output slurm-%j-test-cifar-pgd-rand-34.out
#SBATCH --output slurm-%j-test-pgd-rand-35-fix-order-1235-aggmo.out
## Command(s) to run:
eval "$(conda shell.bash hook)"
conda activate base
# nvidia-smi
# python test.py configs/test_img_rand.yml
# python train.py configs/train_img_pgd-rand.yml
# python train.py configs/train_img_pgd.yml
# python train.py configs/train_cifar_pgd-rand.yml
# python test.py configs/test_cifar_rand.yml
# python test.py configs/test_img_rand_normal.yml
python test.py configs/test_img_rand_fixed.yml