import os
import atexit
import json
import torch
import torch.nn  as nn
import torch.distributed as dist

from torch.utils.data import DistributedSampler
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP

from ..data.dataloader import DataLoader

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'WORLD_SIZE' in os.environ and os.environ['WORLD_SIZE'] != '': # 'RANK' in os.environ and 
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = args.local_rank = int(os.environ['LOCAL_RANK'])
        # local_world_size = int(os.environ['WORLD_SIZE'])
        # args.world_size = args.world_size * local_world_size
        # args.gpu = args.local_rank = int(os.environ['LOCAL_RANK'])
        # args.rank = args.rank * local_world_size + args.local_rank
        # print('world size: {}, rank: {}, local rank: {}'.format(args.world_size, args.rank, args.local_rank))
        # print(json.dumps(dict(os.environ), indent=2))
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.local_rank = int(os.environ['SLURM_LOCALID'])
        args.world_size = int(os.environ['SLURM_NPROCS'])
        
        # print('world size: {}, world rank: {}, local rank: {}, device_count: {}'.format(args.world_size, args.rank, args.local_rank, torch.cuda.device_count()))
        # print("os.environ['SLURM_JOB_NODELIST']:", os.environ['SLURM_JOB_NODELIST'])
        # print(json.dumps(dict(os.environ), indent=2))
        # print('args:')
        # print(json.dumps(vars(args), indent=2))
    else:
        print('Not using distributed mode')
        args.distributed = False
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0
        return

    print("world_size:{} rank:{} local_rank:{}".format(args.world_size, args.rank, args.local_rank))
    args.distributed = True
    torch.cuda.set_device(args.local_rank)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    print("Before torch.distributed.barrier()")
    torch.distributed.barrier()
    print("End torch.distributed.barrier()")
    setup_for_distributed(args.rank == 0)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def warp_loader(loader, shuffle=False):
    if is_dist_avail_and_initialized():
        sampler = DistributedSampler(loader.dataset, shuffle=shuffle)
        loader = DataLoader(loader.dataset,
                            loader.batch_size,
                            sampler=sampler,
                            drop_last=loader.drop_last,
                            collate_fn=loader.collate_fn,
                            pin_memory=loader.pin_memory,
                            num_workers=loader.num_workers)
    return loader


def warp_model(
    model: torch.nn.Module,
    sync_bn: bool=False,
    dist_mode: str='ddp',
    find_unused_parameters: bool=False,
    compile: bool=False,
    compile_mode: str='reduce-overhead',
    **kwargs
):
    if is_dist_avail_and_initialized():
        rank = get_rank()
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model) if sync_bn else model
        if dist_mode == 'dp':
            model = DP(model, device_ids=[rank], output_device=rank)
        elif dist_mode == 'ddp':
            model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=find_unused_parameters)
        else:
            raise AttributeError('')

    if compile:
        model = torch.compile(model, mode=compile_mode)

    return model

@atexit.register
def cleanup():
    """cleanup distributed environment"""
    if is_dist_avail_and_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def is_parallel(model) -> bool:
    # Returns True if model is of type DP or DDP
    return type(model) in (
        torch.nn.parallel.DataParallel,
        torch.nn.parallel.DistributedDataParallel,
    )


def de_parallel(model) -> nn.Module:
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model
