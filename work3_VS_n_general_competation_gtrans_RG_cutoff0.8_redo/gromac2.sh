#!/bin/bash

#SBATCH -J test                     #任务名
#SBATCH -p lzhgnormal               #队列名
#SBATCH -N 1                        #节点数
#SBATCH --ntasks-per-node=2         #每个节点运行任务数
#SBATCH --cpus-per-task=2           #表示每个任务占用4个处理器核
#SBATCH --gres=gpu:1                #每个节点使用的gpu卡数
#SBACH -o %j.o
#SBATCH -e %j.e
source ~/.bashrc
module purge
module load apps/cuda/10.1
conda activate GraphDTA

python  -u  training_nn3_affinity.py




#mpirun -np 10      gmx_mpi mdrun -s npt2.tpr -cpi state.cpt

