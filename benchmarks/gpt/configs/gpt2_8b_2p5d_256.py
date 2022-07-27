import torch
from colossalai.amp import AMP_TYPE
from titans.loss.lm_loss import GPTLMLoss
from titans.model.gpt import gpt2_8B, gpt2_xl
from torch.optim import Adam

VOCAB_SIZE = 50304
SEQ_LENGTH = 1024

TOTAL_BATCH_SIZE = 1280
LEARNING_RATE = 0.00015
WEIGHT_DECAY = 1e-2

TENSOR_PARALLEL_SIZE = 16
TENSOR_PARALLEL_MODE = '2.5d'
NUM_MICRO_BATCHES = 8

NUM_EPOCHS = 60
WARMUP_EPOCHS = int(NUM_EPOCHS * 0.36)

parallel = dict(
    pipeline=2,
    tensor=dict(mode=TENSOR_PARALLEL_MODE, size=TENSOR_PARALLEL_SIZE, depth=1),
)

optimizer = dict(
    type=Adam,
    lr=0.00015,
    weight_decay=1e-2,
)

loss = dict(
    type=GPTLMLoss,
)

fp16 = dict(
    mode=AMP_TYPE.NAIVE
)

model = dict(type=gpt2_8B,
             vocab_size=VOCAB_SIZE,
             max_position_embeddings=SEQ_LENGTH,
             dtype=torch.half,
             fuse_scale_mask_softmax=True,
             checkpoint=True)

fp16 = dict(mode=AMP_TYPE.NAIVE)

gradient_accumulation = 1

BATCH_SIZE = TOTAL_BATCH_SIZE // gradient_accumulation

clip_grad_norm = 1.0

LOG_PATH = f"./gpt2_{TENSOR_PARALLEL_MODE}_tp{TENSOR_PARALLEL_SIZE}_bs{BATCH_SIZE}_lr{LEARNING_RATE}_accum{gradient_accumulation}_clip_grad{clip_grad_norm}/"
