import os
import random
from datetime import timedelta

import torch
import torch.distributed as dist


def initialize(seed):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = (
        torch.device("cuda", local_rank)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(
            "nccl" if device.type == "cuda" else "gloo", timeout=timedelta(hours=2)
        )
    random.seed(seed)
    torch.manual_seed(seed)
    return device, rank, world


def sync_gradients(parameters, world):
    if world > 1:
        for parameter in parameters:
            if parameter.grad is not None:
                dist.all_reduce(parameter.grad)
                parameter.grad.div_(world)


def barrier():
    if dist.is_initialized():
        dist.barrier()


def local_batch_sizes(train, world):
    sizes = train["batch_sizes"]
    if any(value % world or value < world for value in sizes.values()):
        raise ValueError(
            f"Global batch sizes {sizes} must be positive multiples of world_size={world}"
        )
    return {key: value // world for key, value in sizes.items()}
