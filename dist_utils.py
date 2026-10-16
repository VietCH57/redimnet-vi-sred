'''
Small helpers for multi-GPU training with torchrun (DistributedDataParallel, one process per GPU).

    torchrun --standalone --nproc_per_node 2 train.py --config configs/ft_redimnet_m.yaml

Without torchrun (WORLD_SIZE unset) everything degrades to a single process: rank 0, world size 1.
'''

import datetime, os
import torch
import torch.distributed as dist


def init():
    '''Initialise the process group if launched by torchrun. Returns (rank, world_size, local_rank).'''
    ws = int(os.environ.get('WORLD_SIZE', '1'))
    if ws <= 1:
        return 0, 1, 0
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        backend = 'nccl'
    else:
        backend = 'gloo'
    # long timeout: rank 0 alone writes checkpoints / computes metrics while the others wait
    dist.init_process_group(backend, timeout=datetime.timedelta(hours=3))
    return rank, ws, local


def is_dist():
    return dist.is_available() and dist.is_initialized()


def rank():
    return dist.get_rank() if is_dist() else 0


def world():
    return dist.get_world_size() if is_dist() else 1


def is_main():
    return rank() == 0


def barrier():
    if is_dist():
        if dist.get_backend() == 'nccl':
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def broadcast_object(obj):
    '''Value of `obj` on rank 0, returned on every rank.'''
    if not is_dist():
        return obj
    box = [obj]
    dist.broadcast_object_list(box, src=0)
    return box[0]


def gather_objects(obj):
    '''List with the `obj` of every rank (on every rank).'''
    if not is_dist():
        return [obj]
    out = [None] * world()
    dist.all_gather_object(out, obj)
    return out


def all_reduce_sum(t):
    if is_dist():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def cleanup():
    if is_dist():
        dist.destroy_process_group()
