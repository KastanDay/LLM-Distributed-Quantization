import torch
from colossalai.amp import AMP_TYPE
from titans.model.gpt import gpt2_8B, gpt2_large, gpt2_medium, gpt2_xl
from titans.model.quant_gpt import quant_gpt2_8B, quant_gpt2_xl
from torch.optim import Adam

VOCAB_SIZE = 50304
SEQ_LENGTH = 1024

TOTAL_BATCH_SIZE = 32
LEARNING_RATE = 0.00015
WEIGHT_DECAY = 1e-2

TENSOR_PARALLEL_SIZE = 2  ## CHANGED THIS
TENSOR_PARALLEL_MODE = '1d'

NUM_EPOCHS = 3
WARMUP_EPOCHS = 1

parallel = dict(
    pipeline=2,
    tensor=dict(mode=TENSOR_PARALLEL_MODE, size=TENSOR_PARALLEL_SIZE),
)

model_dtypes = dict(embed_dtype=torch.float16, head_dtype=torch.float16, layernorm_dtype=torch.float16, decoder_dtype=torch.float16)

model = dict(type=quant_gpt2_xl,
            **model_dtypes,          # NOTE custon data types
             vocab_size=VOCAB_SIZE,
             max_position_embeddings=SEQ_LENGTH,
            #  fuse_scale_mask_softmax=True,
             checkpoint=False)

optimizer = dict(
    type=Adam,
    lr=0.00015,
    weight_decay=1e-2,
)

fp16 = dict(mode=AMP_TYPE.NAIVE)

gradient_accumulation = 1

BATCH_SIZE = TOTAL_BATCH_SIZE // gradient_accumulation

clip_grad_norm = 1.0

# LOG_PATH = f"./gpt2_{TENSOR_PARALLEL_MODE}_tp{TENSOR_PARALLEL_SIZE}_bs{BATCH_SIZE}_lr{LEARNING_RATE}_accum{gradient_accumulation}_clip_grad{clip_grad_norm}/"
