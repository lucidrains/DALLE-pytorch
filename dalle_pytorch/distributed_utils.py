"""
Utility functions for optional distributed execution.

To use,
1. set the `BACKENDS` to the ones you want to make available,
2. in the script, wrap the argument parser with `wrap_arg_parser`,
3. in the script, set and use the backend by calling
   `set_backend_from_args`.

You can check whether a backend is in use with the `using_backend`
function.
"""

from torch.nn import Module

from dalle_pytorch.distributed_backends import \
    DeepSpeedBackend, \
    DummyBackend, \
    HorovodBackend

_DEFAULT_BACKEND = DummyBackend()
"""Which backend to use by default. Assumed to be _not_ distributed."""

BACKENDS = [
    _DEFAULT_BACKEND,
    DeepSpeedBackend(),
    HorovodBackend(),
]

is_distributed = None
"""Whether we are distributed."""
backend = None
"""Backend in usage."""


class ZeROLoading():
    def __enter__(self):
        if hasattr(Module, '_old_load_state_dict'):
            return
        Module._old_load_state_dict = Module.load_state_dict
        Module.load_state_dict = load_checkpoint


    def __exit__(self, exc_type, exc_value, traceback):
        if not hasattr(Module, '_old_load_state_dict'):
            return
        Module.load_state_dict = Module._old_load_state_dict
        del Module._old_load_state_dict

        if exc_type is not None:
            return False


def wrap_arg_parser(parser):
    """Add arguments to support optional distributed backend usage."""
    parser.add_argument(
        '--distributed_backend',
        '--distr_backend',
        type=str,
        default=None,
        help='which distributed backend to use. Do not distribute by default',
    )
    for distr_backend in BACKENDS:
        parser = distr_backend.wrap_arg_parser(parser)
    return parser


def set_backend_from_args(args):
    """Set and return the backend based on the given `args`."""
    global is_distributed, backend

    # Handle this specially for backwards compatibility.
    if args.deepspeed:
        args.distributed_backend = DeepSpeedBackend.BACKEND_NAME

    if not args.distributed_backend:
        is_distributed = False
        backend = _DEFAULT_BACKEND
        return backend

    backend_name = args.distributed_backend.lower()
    for distr_backend in BACKENDS:
        if distr_backend.BACKEND_NAME.lower() == backend_name:
            backend = distr_backend
            if not backend.has_backend():
                raise ModuleNotFoundError(
                    f'{backend.BACKEND_NAME} backend selected but '
                    'module not available'
                )

            print(f'Using {backend.BACKEND_NAME} for distributed execution')
            is_distributed = True
            return backend

    raise ValueError(
        'unknown backend; please check `distributed_utils.BACKENDS`')


def require_set_backend():
    """Raise an `AssertionError` when the backend has not been set."""
    assert backend is not None, (
        'distributed backend is not set. Please call '
        '`distributed_utils.set_backend_from_args` at the start of your script'
    )


def using_backend(test_backend):
    """Return whether the backend is set to `test_backend`.

    `test_backend` may be a string of the name of the backend or
    its class.
    """
    require_set_backend()
    if isinstance(test_backend, str):
        return backend.BACKEND_NAME == test_backend
    return isinstance(backend, test_backend)


def load_checkpoint(model, state_dict, strict=True):
    if is_distributed and using_backend(DeepSpeedBackend):
        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        metadata = getattr(state_dict, '_metadata', None)

        def load_partitioned(module, prefix=''):
            if metadata is None:
                local_metadata = {}
            else:
                local_metadata = metadata.get(prefix[:-1], {})

            with backend.backend_module.zero.GatheredParameters(
                    list(module.parameters(recurse=False)),
                    modifier_rank=backend.ROOT_RANK,
            ):
                if backend.is_root_worker():
                    module._load_from_state_dict(
                        state_dict,
                        prefix,
                        local_metadata,
                        True,
                        missing_keys,
                        unexpected_keys,
                        error_msgs,
                    )

            for name, child in module._modules.items():
                if child is not None:
                    load_partitioned(child, prefix + name + '.')

        load_partitioned(model)
    else:
        model._old_load_state_dict(state_dict, strict)
