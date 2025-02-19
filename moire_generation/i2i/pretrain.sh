#!/bin/bash -e
#SBATCH --job-name=moire-pretrain
#SBATCH --partition=a10
#SBATCH --nodelist=aten230
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=a10:8
#SBATCH --output=./logs/fas-moire-%j.out

echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "Number of GPUs allocated to the batch step:=" $SLURM_GPUS_ON_NODE
echo "Requested GPU count per allocated node:=" $SLURM_GPUS_PER_NODE
echo "Number of GPUs requested:=" $SLURM_GPUS
echo "global GPU IDs of the GPUs allocated to this job:=" $SLURM_JOB_GPUS
echo "Slurm task ID:=" $SLURM_PROCID
echo "The process ID of the task being starte:=" $SLURM_TASK_PID
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "


echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "$MASTER_ADDR"

srun -K --container-name=pyxis_face \
    --container-image=/purestorage/project/syshin/enroot_image/facepuzzle_infer.sqsh \
    --container-mounts /purestorage:/purestorage,/home/tyk:/workspace --container-workdir /purestorage/project/tyk/3_CUProjects/FAS/flimgan --container-writable \
    bash -c 'echo "$(hostname), $(nvidia-smi -L), $SLURM_PROCID,$SLURM_JOB_ID,$SLURM_TASK_PID" && torchrun --standalone --nnodes 1 --nproc_per_node 8 --rdzv_id $RANDOM --rdzv_backend static --master_addr $MASTER_ADDR --master_port 8882 --node_rank $SLURM_PROCID pretrain.py'


echo "All jobs completed!"
echo ""
echo "################################################################"
echo "@@@@@@@@@@@@@@@@@@ Run completed at:- @@@@@@@@@@@@@@@@@@@@@@@@@"
date
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "################################################################"