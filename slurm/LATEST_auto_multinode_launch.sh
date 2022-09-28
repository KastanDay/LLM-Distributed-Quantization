#!/bin/bash -l

#SBATCH --job-name=pdg_viz_gpu
#SBATCH --partition=gpuA40x4
#SBATCH --account=<tbd>

#SBATCH --time=4:00:00
#SBATCH --nodes=8
#SBATCH --ntasks=8

#SBATCH --gpus-per-node=2
#SBATCH --mem=0
#SBATCH --exclusive

echo "Launcher: LATEST_auto_multinode_launch.sh"
###############################
## 😄 ~ Configure me !! 👈 ~~ ## 
###############################
export PROJECT_BASE_DIR=$HOME/LLM-Distributed-Quantization
export EXPERIMENT_BASE_DIR=${PROJECT_BASE_DIR}/benchmarks/gpt

export TRAIN_FILEPATH=${EXPERIMENT_BASE_DIR}/v2_train.py
export DATA=${PROJECT_BASE_DIR}/datasets/small-gpt-dataset.json
# export CONFIG_FILEPATH=${PROJECT_BASE_DIR}/benchmarks/gpt/configs/gpt2_8b_2p5d_256.py

# tags for logs, like precision or parallel (easy grouping in the charts)
export EXPERIMENT_START_TIME=$(date +"%h-%d__%H:%M")
export WANDB_MODE=online  # or offline, when more than 32 GPUs. It will save logs to CWD from wherever the shell is. 
###############################

# less frequent config settings:
export CONDA_ENV_NAME=col_ai_quant
export NUM_GPUS_PER_NODE=1
export WANDB_SLURM_ID=$SLURM_JOB_ID
export WANDB_SLURM_ID=$SLURM_JOB_ID
MAIN_HOST_PORT=29500


## PARAMS ##
# pass config as first param param
if [[ -z $1 ]];
then 
    export CONFIG_FILEPATH=$EXPERIMENT_BASE_DIR/configs/quant_gpt2_single_gpu.py
    echo "⚠️ Missing FIRST parameter, using DEFAULT config $CONFIG_FILEPATH ⚠️" 
else
    export CONFIG_FILEPATH=$EXPERIMENT_BASE_DIR/configs/$1
    echo "Param 1: CONFIG_FILEPATH = $CONFIG_FILEPATH"
fi
# saving 2nd param code for later.
# if [[ -z $2 ]];
# then 
#     export NUM_NODES_REQUEST=2
#     echo "⚠️ Missing FIRST parameter, using DEFAULT num_nodes $NUM_NODES_REQUEST ⚠️" 
# else
#     export NUM_NODES_REQUEST=$2
#     echo "Param 2: NUM_NODES = $NUM_NODES_REQUEST"
# fi


### Starting script ###
echo "Loding cuda (module load cuda/11.7.0)..."
module load cuda/11.7.0

conda activate $CONDA_ENV_NAME
sleep 0.25
echo "done activating $CONDA_ENV_NAME"

#### Slurm cluster info ####
# find head-node
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
export MAIN_HOST=${nodes_array[0]}
export WORLD_SIZE=$(($NUM_GPUS_PER_NODE * $SLURM_JOB_NUM_NODES))
export WANDB_WORLD_SIZE=$(($NUM_GPUS_PER_NODE * $SLURM_JOB_NUM_NODES))

echo "Starting on hostname: $(hostname | cut -c 1-7)"
echo "  JobID:= " $SLURM_JOB_ID
echo "  Nodelist:= " $SLURM_JOB_NODELIST
echo "  Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "  GPUs per node:= " $SLURM_GPUS_ON_NODE
echo "  CPUs per node:= " $SLURM_CPUS_ON_NODE
echo "  NTasks per node:= "  $SLURM_NTASKS_PER_NODE
# SLURM_NPROCS is my "world size", meaning total # of (GPU) devices.
echo "  Slurm NPROCS:= "  $SLURM_NPROCS
echo "World size: $WORLD_SIZE"


# todo pass in the date_time from here (so they're all the same for grouping!)
# even minutes is too short... 

# todo launch via torchrun. 

# the total numbner of workes is worker_num=$SLURM_JOB_NUM_NODES
localrank=0
for ((node_i = 0; node_i < $SLURM_JOB_NUM_NODES; node_i++)); do
    local_node_hostname=${nodes_array[$node_i]}

    for ((gpu_i = 0; gpu_i < $NUM_GPUS_PER_NODE; gpu_i++)); do
        echo "Starting GPU worker rank $localrank at $local_node_hostname"

        # localrank=$((gpu_i + node_i * $NUM_GPUS_PER_NODE)) # smarter way, but what I have is fine.
        
        #######################
        #### Main Launcher ####
        #######################
        # srun --nodes=1 --ntasks=1 -w "$local_node_hostname" \
        # TODO: add `wandb offline;` when necessary.
        ssh "$local_node_hostname" \
            "conda activate $CONDA_ENV_NAME; export DATA=$DATA; wandb online" \  
            "python $TRAIN_FILEPATH --config $CONFIG_FILEPATH --host $MAIN_HOST --port $MAIN_HOST_PORT --world_size $WORLD_SIZE --rank $localrank" &
        
        # monotonically incrememnt local rank by 1 for EVERY gpu launched
        ((localrank=localrank+1))
        sleep 0.1
    done
done

echo "Done launching GPU workers."
echo "NOTE: Only when sufficient workers have connected to the head node will the training automatically begin."
sleep infinity

