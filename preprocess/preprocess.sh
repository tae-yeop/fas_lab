#!/bin/bash -l

##SBATCH --nodelist=nv180

#SBATCH --time=99:00:00
#SBATCH -p 40g
#SBATCH --nodes=1            # This needs to match Trainer(num_nodes=...)
#SBATCH --ntasks-per-node=1  # This needs to match Trainer(devices=...)
#SBATCH --mem=64gb
#SBATCH --cpus-per-task=32
#SBATCH -o ./logs/%A.txt

# /purestorage/project/shhong/enroot_images/pytorch_2_5_1.sqsh
srun --container-image /purestorage/project/tyk/0_Software/test.sqsh \
    --container-mounts /purestorage:/purestorage,/purestorage/project/tyk/0_Software/cache:/home/$USER/.cache \
    --no-container-mount-home --unbuffered \
    --container-writable \
    --container-workdir /purestorage/project/tyk/3_CUProjects/iBeta \
    bash -c "
    python preprocess_lm.py $@
    "