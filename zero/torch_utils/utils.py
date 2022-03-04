import os

import torch
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP


def init_w_torch(builder, config):
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    host = os.environ['MASTER_ADDR']
    port = int(os.environ['MASTER_PORT'])
    init_process_group(rank=rank, world_size=world_size, init_method=f'tcp://{host}:{port}', backend='nccl')

    torch.cuda.set_device(rank)

    build_data, build_model, build_loss, build_optimizer, build_scheduler = builder()

    train_data, test_data = build_data()

    model = build_model().to(rank)
    model = DDP(model)

    criterion = build_loss()

    optimizer = build_optimizer(model.parameters())

    scaler = torch.cuda.amp.GradScaler(**config['mixed_precision']) if 'mixed_precision' in config else None

    lr_scheduler = build_scheduler(len(train_data), optimizer)

    return model, train_data, test_data, criterion, optimizer, scaler, lr_scheduler