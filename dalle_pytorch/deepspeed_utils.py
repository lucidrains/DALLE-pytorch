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
            action=store_true,
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


def maybe_init_deepspeed(
        args=None,
        model=None,
        optimizer=None,
        model_parameters=None,
        training_data=None,
        lr_scheduler=None,
        **kwargs,
):
    """Return a model engine, optimizer, dataloader, and learning rate
    scheduler. These are obtained by importing and initializing the
    given values on DeepSpeed, depending on `args`.
    If DeepSpeed is not desired, return them unchanged.

    For the other arguments, see `deepspeed.initialize`.
    """
    if not args.deepspeed:
        return (model, optimizer, training_data, lr_scheduler)

    return deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        model_parameters=model_parameters,
        training_data=training_data,
        **kwargs,
    )
