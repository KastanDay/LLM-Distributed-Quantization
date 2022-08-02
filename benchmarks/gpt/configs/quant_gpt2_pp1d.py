import torch
from colossalai.amp import AMP_TYPE
from colossalai.engine.schedule import PipelineSchedule
from titans.model.gpt import gpt2_8B_pipeline
from titans.model.quant_gpt import quant_gpt2_8B, quant_gpt2_xl
from torch.optim import Adam

VOCAB_SIZE = 50304
SEQ_LENGTH = 1024

TOTAL_BATCH_SIZE = 64
BATCH_SIZE = TOTAL_BATCH_SIZE #// gradient_accumulation
LEARNING_RATE = 0.00015
WEIGHT_DECAY = 1e-2

TENSOR_PARALLEL_SIZE = 4
TENSOR_PARALLEL_MODE = '1d'

PIPELINE_SIZE = 2
MICRO_BATCH_SIZE = 4
NUM_MICRO_BATCHES = BATCH_SIZE // MICRO_BATCH_SIZE

NUM_EPOCHS = 20
WARMUP_EPOCHS = 1

parallel = dict(
    pipeline=PIPELINE_SIZE,
    tensor=dict(mode=TENSOR_PARALLEL_MODE, size=TENSOR_PARALLEL_SIZE),
)

optimizer = dict(
    type=Adam,
    lr=0.00015,
    weight_decay=1e-2,
)

model = dict(type=quant_gpt2_8B,
             vocab_size=VOCAB_SIZE,
             max_position_embeddings=SEQ_LENGTH,
            #  dtype=torch.half,
             checkpoint=False)

schedule = dict(type=PipelineSchedule,
                num_microbatches=NUM_MICRO_BATCHES,
                tensor_shape=(MICRO_BATCH_SIZE, SEQ_LENGTH, 3072),
                scatter_gather_tensors=True)

# fp16 = dict(mode=AMP_TYPE.NAIVE, )
# gradient_accumulation = 1
# clip_grad_norm = 1.0

LOG_PATH = f"./gpt3_{TENSOR_PARALLEL_MODE}_pp{PIPELINE_SIZE}_tp{TENSOR_PARALLEL_SIZE}_bs{BATCH_SIZE}_lr{LEARNING_RATE}_accum/"
