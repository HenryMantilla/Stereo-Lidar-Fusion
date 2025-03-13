#!/bin/bash

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
echo "Node IDs of participating nodes ${nodes_array[*]}"

# Get the IP address and set port for MASTER node
head_node="${nodes_array[0]}"
echo "Getting the IP address of the head node ${head_node}"
#master_ip=$(srun -n 1 -N 1 --gpus=1 -w ${head_node} /bin/hostname -I | cut -d " " -f 2)
master_ip=$(hostname -I | cut -d' ' -f1)
master_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo "head node is ${master_ip}:${master_port}"

# Set environment variables
export OMP_NUM_THREADS=1
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nnodes=1 --nproc_per_node=4 --node_rank=0 --master_addr=${master_ip} --master_port=${master_port} train.py \
  --model depth_fusion_pvt \
  --dataset kitti_completion \
  --data_path Data \
  --crop_size 256 768 \
  --num_workers 3 \
  --seed 123 \
  --optimizer adamw \
  --scheduler cosine \
  --weight_decay 1e-4 \
  --lr 1e-4 \
  --batch_size 3 \
  --epochs 20 \
  --ckpt_dir ./checkpoints/crop_256x1216_lr_1e-3_adamW_tests \
  --save_ckpt_freq 4 \
  --val_freq 4 \
