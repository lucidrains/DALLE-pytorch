import os

import torch

from .distributed_backend import DistributedBackend


class DeepSpeedBackend(DistributedBackend):
    """Distributed backend using the DeepSpeed engine."""

    BACKEND_MODULE_NAME = 'deepspeed'
    BACKEND_NAME = 'DeepSpeed'

    def wrap_arg_parser(self, parser):
        if not self.has_backend():
            parser.add_argument(
                '--deepspeed',
                type=lambda _: False,
                help=(
                    'whether to use DeepSpeed '
                    "(ignored since it's not available)"
                ),
            )
        else:
            parser = self.backend_module.add_config_arguments(parser)

        parser.add_argument(
            '--local_rank',
            type=int,
            default=-1,
            help='local rank passed from distributed launcher',
        )
        return parser

    def _initialize(self):
        self.backend_module.init_distributed()

    @staticmethod
    def _require_torch_distributed_init():
        """Raise an error when `torch.distributed` has not been
        initialized yet.
        """
        assert torch.distributed.is_initialized(), \
            ('`torch.distributed` is not initialized; please call '
             '`DeepSpeedBackend.initialize` at the start of your script')

    def _get_world_size(self):
        self._require_torch_distributed_init()
        return torch.distributed.get_world_size()

    def _get_rank(self):
        self._require_torch_distributed_init()
        return torch.distributed.get_rank()

    def _get_local_rank(self):
        self._require_torch_distributed_init()
        return int(os.environ['LOCAL_RANK'])

    def _local_barrier(self):
        self._require_torch_distributed_init()
        torch.distributed.barrier()

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

        For the other or other possible arguments,
        see `deepspeed.initialize`.
        """
        if not args.deepspeed:
            print(
                'WARNING: DeepSpeed backend was selected; setting '
                '`args.deepspeed = True`'
            )
            args.deepspeed = True

        return self.backend_module.initialize(
            args=args,
            model=model,
            optimizer=optimizer,
            model_parameters=model_parameters,
            training_data=training_data,
            lr_scheduler=lr_scheduler,
            **kwargs,
        )

    def _average_all(self, tensor):
        self._require_torch_distributed_init()
        # We copy because modification happens in-place
        averaged = tensor.detach().clone()
        # We use `all_reduce` because it is better supported than `reduce`
        torch.distributed.all_reduce(averaged, torch.distributed.ReduceOp.SUM)
        return averaged / self.get_world_size()
