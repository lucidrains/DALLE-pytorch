from .distributed_backend import DistributedBackend


class DummyBackend(DistributedBackend):
    """Acts like a distributed backend.

    Used as a stand-in replacement to obtain a non-distributed program.
    """

    # We define this so we can use `super().__init__` but want this to
    # throw an error upon import.
    BACKEND_MODULE_NAME = 'NO MODULE'
    BACKEND_NAME = 'Dummy'

    def has_backend(self):
        return True

    def wrap_arg_parser(self, parser):
        return parser

    def _initialize(self):
        pass

    def _get_world_size(self):
        return 1

    def _get_rank(self):
        return self.ROOT_RANK

    def _get_local_rank(self):
        return self.ROOT_RANK

    def _local_barrier(self):
        pass

    def _distribute(
            self,
            _args=None,
            model=None,
            optimizer=None,
            _model_parameters=None,
            training_data=None,
            lr_scheduler=None,
            **_kwargs,
    ):
        """Return the model, optimizer, dataloader, and learning rate scheduler
        as is.
        """
        return (model, optimizer, training_data, lr_scheduler)

    def _average_all(self, tensor):
        return tensor
