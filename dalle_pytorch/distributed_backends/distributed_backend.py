"""
An abstract backend for distributed deep learning.

Provides several standard utility methods under a common API.
Please check the documentation of the class `DistributedBackend` for
details to implement a new backend.
"""

from importlib import import_module


class DistributedBackend:
    """An abstract backend class for distributed deep learning.

    Provides several standard utility methods under a common API.
    Variables that must be overridden:
    - BACKEND_MODULE_NAME
    - BACKEND_NAME
    Methods that must be overridden:
    - wrap_arg_parser
    - _initialize
    - _get_world_size
    - _get_rank
    - _get_local_rank
    - _local_barrier
    - _distribute
    - _average_all
    """

    BACKEND_MODULE_NAME = None
    """Name of the module to import for the backend."""
    BACKEND_NAME = None
    """Name of the backend for printing."""

    ROOT_RANK = 0

    backend_module = None
    """The module to access the backend."""
    is_initialized = False
    """Whether the backend is initialized."""

    def __init__(self):
        if self.BACKEND_MODULE_NAME is None:
            raise NotImplementedError('BACKEND_MODULE_NAME is not set')
        if self.BACKEND_NAME is None:
            raise NotImplementedError('BACKEND_NAME is not set')

    def has_backend(self):
        """Return whether the backend module is now imported."""
        try:
            self.backend_module = import_module(self.BACKEND_MODULE_NAME)
        except ModuleNotFoundError:
            return False
        return True

    def check_batch_size(self, batch_size):
        """Check whether the batch size makes sense for distribution."""
        assert batch_size >= self.get_world_size(), \
            (f"batch size can't be smaller than number of processes "
             f'({batch_size} < {self.get_world_size()})')

    def wrap_arg_parser(self, parser):
        """Add arguments to support optional distributed backend usage."""
        raise NotImplementedError

    def initialize(self):
        """Initialize the distributed backend."""
        self._initialize()
        self.is_initialized = True

    def _initialize(self):
        """Initialize the distributed backend."""
        raise NotImplementedError

    def require_init(self):
        """Raise an error when the backend has not been initialized yet."""
        assert self.is_initialized, \
            (f'{BACKEND_NAME} backend has not been initialized; please call '
             f'`distributed_utils.initialize` at the start of your script to '
             f'allow optional distributed usage')

    def get_world_size(self):
        """Return the amount of distributed processes."""
        self.require_init()
        return self._get_world_size()

    def _get_world_size(self):
        """Return the amount of distributed processes."""
        raise NotImplementedError

    def get_rank(self):
        """Return the global rank of the calling worker process."""
        self.require_init()
        return self._get_rank()

    def _get_rank(self):
        """Return the global rank of the calling worker process."""
        raise NotImplementedError

    def get_local_rank(self):
        """Return the local rank of the calling worker process.
        The local rank is the rank based on a single node's processes.
        """
        self.require_init()
        return self._get_local_rank()

    def _get_local_rank(self):
        """Return the local rank of the calling worker process.
        The local rank is the rank based on a single node's processes.
        """
        raise NotImplementedError

    def is_root_worker(self):
        """Return whether the calling worker has the root rank."""
        return self.get_rank() == self.ROOT_RANK

    def is_local_root_worker(self):
        """Return whether the calling worker has the root rank on this node."""
        return self.get_local_rank() == self.ROOT_RANK

    def local_barrier(self):
        """Wait until all processes on this node have called this function."""
        self.require_init()
        self._local_barrier()

    def _local_barrier(self):
        """Wait until all processes on this node have called this function."""
        raise NotImplementedError

    def distribute(
            self,
            args=None,
            model=None,
            optimizer=None,
            model_parameters=None,
            training_data=None,
            lr_scheduler=None,
            **kwargs,
    ):
        """Return a distributed model engine, optimizer, dataloader, and
        learning rate scheduler. These are obtained by wrapping the
        given values with the backend.
        """
        self.require_init()
        return self._distribute(
            args,
            model,
            optimizer,
            model_parameters,
            training_data,
            lr_scheduler,
            **kwargs,
        )

    def _distribute(
            self,
            args=None,
            model=None,
            optimizer=None,
            model_parameters=None,
            training_data=None,
            lr_scheduler=None,
            **kwargs,
    ):
        """Return a distributed model engine, optimizer, dataloader, and
        learning rate scheduler. These are obtained by wrapping the
        given values with the backend.
        """
        raise NotImplementedError

    def average_all(self, tensor):
        """Return the average of `tensor` over all workers."""
        self.require_init()
        return self._average_all(tensor)

    def _average_all(self, tensor):
        """Return the average of `tensor` over all workers."""
        raise NotImplementedError
