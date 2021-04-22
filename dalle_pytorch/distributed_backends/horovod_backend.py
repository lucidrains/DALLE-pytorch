import torch

from .distributed_backend import DistributedBackend


class HorovodBackend(DistributedBackend):
    """Distributed backend using Horovod."""

    BACKEND_MODULE_NAME = 'horovod.torch'
    BACKEND_NAME = 'Horovod'

    def wrap_arg_parser(self, parser):
        return parser

    def check_batch_size(self, batch_size):
        # Horovod uses the local batch size to determine the effective
        # batch size.
        pass

    def _initialize(self):
        self.backend_module.init()
        if torch.cuda.is_available():
            torch.cuda.set_device(self._get_local_rank())

    def _get_world_size(self):
        return self.backend_module.size()

    def _get_rank(self):
        return self.backend_module.rank()

    def _get_local_rank(self):
        return self.backend_module.local_rank()

    def _local_barrier(self):
        # Actually a global barrier but works for our purposes.
        self.backend_module.join()

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
        optimizer = self.backend_module.DistributedOptimizer(optimizer)
        self.backend_module.broadcast_parameters(
            model.state_dict(), root_rank=self.ROOT_RANK)
        self.backend_module.broadcast_optimizer_state(
            optimizer, root_rank=self.ROOT_RANK)
        return (model, optimizer, training_data, lr_scheduler)

    def _average_all(self, tensor):
        # Reduce op is average by default
        averaged = self.backend_module.allreduce(tensor)
        return averaged
