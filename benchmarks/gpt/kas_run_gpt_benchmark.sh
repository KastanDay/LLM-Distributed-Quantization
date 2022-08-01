NUM_GPUS_PER_NODE=4
NUM_NODES=1
NODE_RANK=0

export CONFIG="./configs/gpt2_2d.py"

export EXEC="torchrun"

BASE_DIR=/u/kastanday/LLM-Distributed-Quantization
export DATA=${BASE_DIR}/datasets/small-gpt-dataset.json
export LOG_PATH=${BASE_DIR}/benchmarks/gpt/logs

${EXEC} --nproc_per_node=${NUM_GPUS_PER_NODE} \
                                --nnodes=${NUM_NODES} \
                                --node_rank=${NODE_RANK} \
                                train.py --from_torch \
                                --config ${CONFIG} \
                                # --wandb_tags first_wandb_tag