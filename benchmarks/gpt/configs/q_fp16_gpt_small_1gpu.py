import torch
# from colossalai.amp import AMP_TYPE
# from titans.model.gpt import gpt2_8B, gpt2_large, gpt2_medium, gpt2_xl
from titans.model.quant_gpt import quant_gpt2_small
from torch.optim import Adam

VOCAB_SIZE = 50304
SEQ_LENGTH = 256

TOTAL_BATCH_SIZE = 48
LEARNING_RATE = 0.00015
WEIGHT_DECAY = 1e-2

# TENSOR_PARALLEL_SIZE = 1  ## CHANGED THIS
# TENSOR_PARALLEL_MODE = '1d'

NUM_EPOCHS = 2
WARMUP_EPOCHS = 1

parallel = dict(
    pipeline=1,
    # tensor=dict(mode=TENSOR_PARALLEL_MODE, size=TENSOR_PARALLEL_SIZE),
)

# To specify dtypes, just edit them here.
# model_dtypes = dict(embed_dtype=torch.bfloat16, head_dtype=torch.bfloat16, layernorm_dtype=torch.bfloat16, decoder_dtype=torch.bfloat16)
model_dtypes = dict(embed_dtype=torch.float16, head_dtype=torch.float16, layernorm_dtype=torch.float16, decoder_dtype=torch.float16)
# model_dtypes = dict(embed_dtype=torch.float32, head_dtype=torch.float32, layernorm_dtype=torch.float32, decoder_dtype=torch.float32)

model = dict(type=quant_gpt2_small,
             **model_dtypes,          # NOTE custon data types
             vocab_size=VOCAB_SIZE,
             max_position_embeddings=SEQ_LENGTH,
            #  dtype=torch.half,
            #  fuse_scale_mask_softmax=True,
             checkpoint=False)

optimizer = dict(
    type=Adam,
    lr=0.00015,
    weight_decay=1e-2,
)

# schedule = dict(type=PipelineSchedule,
#                 num_microbatches=NUM_MICRO_BATCHES,
#                 tensor_shape=(MICRO_BATCH_SIZE, SEQ_LENGTH, 3072),
#                 scatter_gather_tensors=True)

# fp16 = dict(mode=AMP_TYPE.NAIVE)

# gradient_accumulation = 1

BATCH_SIZE = TOTAL_BATCH_SIZE #// gradient_accumulation

# clip_grad_norm = 1.0

LOG_PATH = f"./gpt2_bs{BATCH_SIZE}_lr{LEARNING_RATE}/"
