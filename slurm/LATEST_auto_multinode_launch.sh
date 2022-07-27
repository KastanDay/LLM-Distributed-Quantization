#!/bin/bash -l

#SBATCH --job-name=pdg_viz_gpu
#SBATCH --partition=gpuA40x4
#SBATCH --account=${MY_ACCOUNT_GPU}

#SBATCH --time=4:00:00
#SBATCH --nodes=8
#SBATCH --ntasks=8

#SBATCH --gpus-per-task=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16

echo "Starting on hostname: $(hostname | cut -c 1-7)"
echo "  JobID:= " $SLURM_JOB_ID
echo "  Nodelist:= " $SLURM_JOB_NODELIST
echo "  Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "  GPUs per node:= " $SLURM_GPUS_ON_NODE
echo "  CPUs per node:= " $SLURM_CPUS_ON_NODE
# echo "  NTasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "  Slurm NPROCS:= "  $SLURM_NPROCS
# SLURM_NPROCS is my "world size", meaning total # of (GPU) devices.
# export SLURM_NPROCS=(($SLURM_GPUS_ON_NODE * $SLURM_NPROCS))

echo "Loding cuda (module load cuda/11.7.0)..."
module load cuda/11.7.0

conda activate col_ai_old_v5
sleep 1
echo "done activating col_ai_old_v5"

#### 👉 CHANGE ME 😊 ####
export BASE_DIR=/u/kastanday/LLM-Distributed-Quantization/benchmarks/gpt
export CONFIG_FILEPATH=${BASE_DIR}/configs/gpt2_2d.py
export DATA=${BASE_DIR}/datasets/small-gpt-dataset.json
# export MY_WANDB_TAGS=$(date +"%h-%d__%H:%M")

export NUM_GPUS_PER_NODE=4
export WORLD_SIZE=$(($NUM_GPUS_PER_NODE * $SLURM_JOB_NUM_NODES))
echo "World size: $WORLD_SIZE"

# COLLECT NODE INFO
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
export MAIN_HOST=${nodes_array[0]}

# worker_num=$((SLURM_JOB_NUM_NODES))
localrank=0

for ((node_i = 0; node_i < $SLURM_JOB_NUM_NODES; node_i++)); do
    local_node_hostname=${nodes_array[$node_i]}

    for ((gpu_i = 0; gpu_i < $NUM_GPUS_PER_NODE; gpu_i++)); do
        echo "Starting GPU worker rank $localrank at $local_node_hostname"

        # localrank=$((gpu_i + node_i * $NUM_GPUS_PER_NODE)) # smarter way, but what I have is fine.
        
        ######################
        #### MAIN COMMAND ####
        ######################"
        # srun --nodes=1 --ntasks=1 -w "$local_node_hostname" \
        ssh "$local_node_hostname" \
            "export DATA=/u/kastanday/colossal_ai/raw_json_backup/train_data_FINAL.json; conda activate col_ai_old_v5; python $BASE_DIR/train_gpt.py --config $CONFIG_FILEPATH --host $MAIN_HOST --port 29500 --world_size $WORLD_SIZE --rank $localrank" &
        
        # strictly incrememnt local rank by 1 for each gpu launched
        ((localrank=localrank+1))
        sleep 0.25
    done
done

echo "Done launching GPU workers."
sleep infinity

