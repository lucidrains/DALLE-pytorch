import os

import torch.distributed

ROOT_RANK = 0

using_deepspeed = None


def has_deepspeed():
    """Return whether the DeepSpeed module is now imported."""
    global deepspeed
    try:
        import deepspeed
    except ModuleNotFoundError:
        return False
    return True


def wrap_arg_parser(parser):
    """Add arguments to support optional DeepSpeed usage."""
    if not has_deepspeed():
        parser.add_argument(
            '--deepspeed',
            type=lambda _: False,
            help="whether to use DeepSpeed (ignored since it's not available)",
        )
    else:
        parser = deepspeed.add_config_arguments(parser)

    parser.add_argument(
        '--local_rank',
        type=int,
        required=False,
        default=-1,
        help='local rank passed from distributed launcher',
    )
    return parser


def init_deepspeed(do_init):
    """Initialize the DeepSpeed distributed backend."""
    global using_deepspeed
    using_deepspeed = do_init

    if not do_init:
        return
    deepspeed.init_distributed()


def require_init():
    """Raise an error when DeepSpeed has not been initialized yet."""
    assert using_deepspeed is not None, \
        ('DeepSpeed has not been initialized; please call '
         '`deepspeed_utils.init_deepspeed` at the start of your script')


def require_torch_distributed_init():
    """Raise an error when `torch.distributed` has not been
    initialized yet.
    """
    assert torch.distributed.is_initialized(), \
        ('torch.distributed is not initialized; please call '
         '`deepspeed_utils.init_deepspeed` at the start of your script')


def get_world_size():
    """Return the amount of distributed processes."""
    require_init()
    if not using_deepspeed:
        return 1

    require_torch_distributed_init()
    return torch.distributed.get_world_size()


def get_rank():
    """Return the global rank of the calling worker process."""
    require_init()
    if not using_deepspeed:
        return ROOT_RANK

    require_torch_distributed_init()
    return torch.distributed.get_rank()


def get_local_rank():
    """Return the local rank of the calling worker process.
    The local rank is the rank based on a single node's processes.
    """
    require_init()
    if not using_deepspeed:
        return ROOT_RANK

    require_torch_distributed_init()
    return int(os.environ['LOCAL_RANK'])


def is_root_worker():
    """Return whether the calling worker has the root rank.
    This is always True when DeepSpeed is disabled.
    """
    return get_rank() == ROOT_RANK


def is_local_root_worker():
    """Return whether the calling worker has the root rank on this node.
    This is always True when DeepSpeed is disabled.
    """
    return get_local_rank() == ROOT_RANK


def local_barrier():
    """Wait until all processes on this node have called this function."""
    require_init()
    if not using_deepspeed:
        return

    require_torch_distributed_init()
    torch.distributed.barrier()


def maybe_distribute(
        args=None,
        model=None,
        optimizer=None,
        model_parameters=None,
        training_data=None,
        lr_scheduler=None,
        **kwargs,
):
    """Return a model engine, optimizer, dataloader, and learning rate
    scheduler. These are obtained by wrapping the given values with
    DeepSpeed, depending on whether it is used.
    If DeepSpeed is disabled, return them unchanged.

    For the other or other possible arguments,
    see `deepspeed.initialize`.
    """
    require_init()
    if not using_deepspeed:
        return (model, optimizer, training_data, lr_scheduler)

    return deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        model_parameters=model_parameters,
        training_data=training_data,
        lr_scheduler=lr_scheduler,
        **kwargs,
    )


def check_batch_size(batch_size):
    assert batch_size >= get_world_size(), \
        (f"batch size can't be smaller than number of processes "
         f'({batch_size} < {get_world_size()})')


def average_all(tensor):
    """Return the average of `tensor` over all workers."""
    require_init()
    if not using_deepspeed:
        return tensor

    require_torch_distributed_init()
    # We copy because modification happens in-place
    averaged = tensor.detach().clone()
    # We use `all_reduce` because it is better supported than `reduce`
    torch.distributed.all_reduce(averaged, torch.distributed.ReduceOp.SUM)
    return averaged / get_world_size()
