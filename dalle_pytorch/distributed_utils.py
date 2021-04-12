import os

import torch.distributed

ROOT_RANK = 0

BACKENDS = [
    'deepspeed',
    'horovod',
    'none',
]

using_deepspeed = None
current_backend = None


def has_deepspeed():
    """Return whether the DeepSpeed module is now imported."""
    global deepspeed
    try:
        import deepspeed
    except ModuleNotFoundError:
        return False
    return True


def has_horovod():
    """Return whether the Horovod module is now imported."""
    global hvd
    try:
        import horovod.torch as hvd
    except ModuleNotFoundError:
        return False
    return True


def wrap_arg_parser(parser):
    """Add arguments to support optional DeepSpeed usage."""
    if not has_horovod():
        parser.add_argument(
            '--horovod',
            type=lambda _: False,
            help="whether to use Horovod (ignored since it's not available)",
        )
    else:
        parser.add_argument(
            '--horovod',
            action='store_true',
            help="whether to use Horovod",
        )

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


def backend_from_args(args):
    if args.deepspeed and args.horovod:
        raise ValueError('multiple backends selected; please only choose one')

    if args.deepspeed:
        return 'deepspeed'
    elif args.horovod:
        return 'horovod'
    else:
        return 'none'


def init_deepspeed(backend):
    """Initialize the DeepSpeed distributed backend."""
    global using_deepspeed
    global selected_backend

    if not backend in BACKENDS:
        raise KeyError(
            'unsupported backend; please check `distributed_utils.BACKENDS`')

    do_init = backend != 'none'
    using_deepspeed = do_init
    selected_backend = backend

    if not do_init:
        return
    if selected_backend == 'deepspeed':
        deepspeed.init_distributed()
    elif selected_backend == 'horovod':
        hvd.init()
        if torch.cuda.is_available():
            torch.cuda.set_device(hvd.local_rank())


def require_init():
    """Raise an error when DeepSpeed has not been initialized yet."""
    assert using_deepspeed is not None, \
        ('`deepspeed_utils` have not been initialized; please call '
         '`deepspeed_utils.init_deepspeed` at the start of your script to '
         'allow optional DeepSpeed usage')


def require_torch_distributed_init():
    """Raise an error when `torch.distributed` has not been
    initialized yet.
    """
    assert torch.distributed.is_initialized(), \
        ('`torch.distributed` is not initialized; please call '
         '`deepspeed_utils.init_deepspeed` at the start of your script')


def get_world_size():
    """Return the amount of distributed processes."""
    require_init()
    if not using_deepspeed:
        return 1

    if selected_backend == 'deepspeed':
        require_torch_distributed_init()
        return torch.distributed.get_world_size()
    if selected_backend == 'horovod':
        return hvd.size()


def get_rank():
    """Return the global rank of the calling worker process."""
    require_init()
    if not using_deepspeed:
        return ROOT_RANK

    if selected_backend == 'deepspeed':
        require_torch_distributed_init()
        return torch.distributed.get_rank()
    elif selected_backend == 'horovod':
        return hvd.rank()


def get_local_rank():
    """Return the local rank of the calling worker process.
    The local rank is the rank based on a single node's processes.
    """
    require_init()
    if not using_deepspeed:
        return ROOT_RANK

    if selected_backend == 'deepspeed':
        require_torch_distributed_init()
        return int(os.environ['LOCAL_RANK'])
    elif selected_backend == 'horovod':
        return hvd.local_rank()


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

    if selected_backend == 'deepspeed':
        require_torch_distributed_init()
        torch.distributed.barrier()
    elif selected_backend == 'horovod':
        hvd.join()


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

    if selected_backend == 'deepspeed':
        return deepspeed.initialize(
            args=args,
            model=model,
            optimizer=optimizer,
            model_parameters=model_parameters,
            training_data=training_data,
            lr_scheduler=lr_scheduler,
            **kwargs,
        )
    elif selected_backend == 'horovod':
        optimizer = hvd.DistributedOptimizer(optimizer)
        hvd.broadcast_parameters(model.state_dict(), root_rank=ROOT_RANK)
        hvd.broadcast_optimizer_state(optimizer, root_rank=ROOT_RANK)
        return (model, optimizer, training_data, lr_scheduler)


def check_batch_size(batch_size):
    assert batch_size >= get_world_size(), \
        (f"batch size can't be smaller than number of processes "
         f'({batch_size} < {get_world_size()})')


def average_all(tensor):
    """Return the average of `tensor` over all workers."""
    require_init()
    if not using_deepspeed:
        return tensor

    if selected_backend == 'deepspeed':
        require_torch_distributed_init()
        # We copy because modification happens in-place
        averaged = tensor.detach().clone()
        # We use `all_reduce` because it is better supported than `reduce`
        torch.distributed.all_reduce(averaged, torch.distributed.ReduceOp.SUM)
        return averaged / get_world_size()
    elif selected_backend == 'horovod':
        # Reduce op is average by default
        averaged = hvd.allreduce(tensor)
        return averaged
