# Installing 

```bash
git clone --recurse-submodules git@github.com:KastanDay/LLM-Distributed-Quantization.git
```

Install strict dependencies (on `x64`):
```
conda env create -f ./utilities/environment.yml
```

### Launch slurm jobs
Specify the number of nodes required via `--nnodes=8` in slurm. Also set `--ntasks` to the same number as the number of nodes. 

Specify the config path to use as the first parameter. I recommend using absolute paths. 
```bash
# launch sbatch from login node of slurm cluster 
sbatch LATEST_auto_multinode_launch.sh <config_filepath>
```

### Updating
Subsequently pull new changes using:
```bash
cd LLM-Distributed-Quantization; git pull --recurse-submodules
```

# Experiment Tracking
All experiments are tracked with Weights and Biases. View the live progress here: https://wandb.ai/kastan/LLM-Distributed-Quantization

For transparency, the first couple hundred experiments are saved here: https://wandb.ai/kastan/col_ai 

# Benchmark for Tuning Accuracy and Efficiency

## Overview

The benchmark includes our efforts in using Colossal-AI to train different tasks to achieve SOTA results.
We are interested in both validataion accuracy and training speed, and prefer larger batch size to take advantage of more GPU devices.
For example, we trained vision transformer with batch size 512 on CIFAR10 and 4096 on ImageNet1k, which are basically not used in existing works.
Some of the results in the benchmark trained with 8x A100 are shown below.

| Task       | Model        | Training Time | Top-1 Accuracy |
| ---------- | ------------ | ------------- | -------------- |
| CIFAR10    | [ViT-Lite-7/4](https://arxiv.org/pdf/2104.05704.pdf) | ~ 16 min      | ~ 90.5%        |
| ImageNet1k | ViT-S/16     | ~ 16.5 h      | ~ 74.5%        |

The `train.py` script in each task runs training with the specific configuration script in `configs/` for different parallelisms.
Supported parallelisms include data parallel only (ends with `vanilla`), 1D (ends with `1d`), 2D (ends with `2d`), 2.5D (ends with `2p5d`), 3D (ends with `3d`).

Each configuration scripts basically includes the following elements, taking ImageNet1k task as example:
```
TOTAL_BATCH_SIZE = 4096
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 0.3

NUM_EPOCHS = 300
WARMUP_EPOCHS = 32

# data parallel only
TENSOR_PARALLEL_SIZE = 1    
TENSOR_PARALLEL_MODE = None

# parallelism setting
parallel = dict(
    pipeline=1,
    tensor=dict(mode=TENSOR_PARALLEL_MODE, size=TENSOR_PARALLEL_SIZE),
)

fp16 = dict(mode=AMP_TYPE.TORCH, ) # amp setting

gradient_accumulation = 2 # accumulate 2 steps for gradient update

BATCH_SIZE = TOTAL_BATCH_SIZE // gradient_accumulation # actual batch size for dataloader

clip_grad_norm = 1.0 # clip gradient with norm 1.0
```
Upper case elements are basically what `train.py` needs, and lower case elements are what Colossal-AI needs to initialize the training.

## Usage

To start training, use the following command to run each worker:
```
$ DATA=/path/to/dataset python train.py --world_size=WORLD_SIZE \
                                        --rank=RANK \
                                        --local_rank=LOCAL_RANK \
                                        --host=MASTER_IP_ADDRESS \
                                        --port=MASTER_PORT \
                                        --config=CONFIG_FILE
```
It is also recommended to start training with `torchrun` as:
```
$ DATA=/path/to/dataset torchrun --nproc_per_node=NUM_GPUS_PER_NODE \
                                 --nnodes=NUM_NODES \
                                 --node_rank=NODE_RANK \
                                 --master_addr=MASTER_IP_ADDRESS \
                                 --master_port=MASTER_PORT \
                                 train.py --config=CONFIG_FILE
```