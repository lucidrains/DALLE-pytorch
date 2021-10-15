from .distributed_backend import DistributedBackend


class HivemindBackend(DistributedBackend):
    BACKEND_MODULE_NAME = 'hivemind'
    BACKEND_NAME = 'hivemind'

    def wrap_arg_parser(self, parser):
        parser.add_argument('--initial_peers', type=str, nargs='+', default=None)
        parser.add_argument('--experiment_prefix', type=str, default='dalle')
        parser.add_argument('--batch_size_per_step', type=int, default=None)
        parser.add_argument('--target_batch_size', type=int, default=None)
        parser.add_argument('--target_group_size', type=int, default=None)
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
        args=None,
        model=None,
        optimizer=None,
        _model_parameters=None,
        training_data=None,
        lr_scheduler=None,
        **_kwargs,
    ):
        self.dht = self.backend_module.DHT(initial_peers=args.initial_peers, start=True)
        print(f'DHT visible multiaddrs: {self.dht.get_visible_maddrs()}')
        optimizer = self.backend_module.CollaborativeOptimizer(
            optimizer,
            dht=self.dht,
            prefix=args.experiment_prefix,
            batch_size_per_step=args.batch_size_per_step,
            target_batch_size=args.target_batch_size,
            target_group_size=args.target_group_size,
            verbose=True,
            start=True,
        )
        return model, optimizer, training_data, lr_scheduler

    def _average_all(self, tensor):
        return tensor
