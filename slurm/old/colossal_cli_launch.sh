#!/bin/bash -l

#SBATCH --job-name=pdg_viz_gpu
#SBATCH --partition=gpuA40x4
#SBATCH --account=bbki-delta-gpu
#SBATCH --time=4:00:00


#SBATCH --nodes=2
#SBATCH --gpus-per-task=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=16

#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

echo "Starting on hostname: $(hostname | cut -c 1-7)"

echo "Loding cuda (module load cuda/11.7.0)..."
module load cuda/11.7.0

# echo " "
echo "  JobID:= " $SLURM_JOB_ID
echo "  Nodelist:= " $SLURM_JOB_NODELIST
echo "  Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "  GPUs per node:= " $SLURM_GPUS_ON_NODE
echo "  CPUs per node:= " $SLURM_CPUS_ON_NODE
# echo "  NTasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "  (this used to be broken, but is critical) Slurm NPROCS:= "  $SLURM_NPROCS
# SLURM_NPROCS is my "world size", meaning total # of (GPU) devices.
# export SLURM_NPROCS=(($SLURM_GPUS_ON_NODE * $SLURM_NPROCS))
echo "  POST MANUAL EXPORT: Slurm NPROCS:= "  $SLURM_NPROCS

export DATA=~/colossal_ai/raw_json_backup/train_data_FINAL.json
export BASE_DIR=/u/kastanday/new_colossal_ai/ColossalAI/examples/language/gpt

conda activate col_ai_old_v5
echo "done activating col_ai_v4"

echo "sleeping 1 secs, waiting for nodes to be ready..."
sleep 1


echo "writing hostfile (to: $BASE_DIR/nodelist.txt)..."
python $BASE_DIR/slurm/write_hostfile.py
sleep 2

# LAUNCH FROM SLURM
echo "Starting launch from Colossal-CLI Launcher..."
# --master_addr $(hostname -i | head -1) 
# Args:
    # config (Union[str, dict, Config]): Config file or config file path are both acceptable
    # host (str): The master address for distributed training
    # port (str): The master port for distributed training
    # backend (str, optional): Backend for ``torch.distributed``, defaults to ``nccl``
    # seed (int, optional): Specified random seed for every process. Defaults to 1024.
    # verbose (bool, optional): Whether to print logs. Defaults to True.
colossalai run --nproc_per_node $SLURM_GPUS_ON_NODE --hostfile $BASE_DIR/nodelist.txt --master_port 29500 $BASE_DIR/train_gpt.py --config $BASE_DIR/gpt2_configs/gpt2_pp1d_8gpu.py

# colossalai run -host 172.28.23.93 --nproc_per_node=4 train_gpt.py --config gpt2_configs/gpt2_zero3_pp1d.py --from_torch

echo "Finished train_gpt.py. Exiting."