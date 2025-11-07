#!/bin/bash
#SBATCH --job-name=vanilla-densenet80-120
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00
#SBATCH --output=/u/tl0463/scratch/Distillation/BAN/run/out/vanilla_densenet80_120_cifar100.out

module purge
source ~/.bashrc
conda activate ban

cd /n/fs/vision-mix/tl0463/Distillation/BAN/knowledge-distillation-pytorch

LOCAL_TMP=/tmp/${USER}_${SLURM_JOB_ID}
mkdir -p "${LOCAL_TMP}"
export TMPDIR="${LOCAL_TMP}"
trap 'rm -rf "${LOCAL_TMP}"' EXIT

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(shuf -i 29500-65000 -n 1)

RUN_NAME="vanilla_densenet80_120_cifar100"
export WANDB_PROJECT=BAN
export WANDB_RUN_NAME=${RUN_NAME}
export WANDB_MODE=online

if [[ $SLURM_PROCID -eq 0 ]]; then
  wandb login 01126ae90da25bae0d86704140ac978cb9fd9c73
fi

mkdir -p /u/tl0463/scratch/Distillation/BAN/run/out

CONFIG_PATH=/u/tl0463/scratch/Distillation/BAN/knowledge-distillation-pytorch/experiments/ban/vanilla_densenet80_120_cifar100/vanilla_densenet80_120_cifar100.json
MODEL_ROOT=/u/tl0463/scratch/Distillation/BAN/models/vanilla
OUTPUT_DIR=${MODEL_ROOT}/${RUN_NAME}
mkdir -p ${OUTPUT_DIR}

srun hostname -s | sort
srun bash -c 'echo "$SLURM_PROCID $(hostname -s): $CUDA_VISIBLE_DEVICES"'

srun python -u train.py \
  --model_dir=${OUTPUT_DIR} \
  --config=${CONFIG_PATH} \
  --enable_wandb true \
  --wandb_mode online \
  --project BAN \
  --experiment ${RUN_NAME} \
  --seed 42 \
  --wandb_tags vanilla cifar100 densenet80_120 epoch200
