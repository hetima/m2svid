import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl
import random

import signal
from packaging import version
from omegaconf import OmegaConf
from PIL import Image
from webdataset import WebDataset
from torchvision.utils import make_grid
from einops import rearrange

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

from sgm.util import instantiate_from_config
from vtdm.logger import setup_logger
from vtdm.callbacks import TextProgressBar


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    # parser.add_argument(
    #     "-n",
    #     "--name",
    #     type=str,
    #     const=True,
    #     default="",
    #     nargs="?",
    #     help="postfix for logdir",
    # )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )

    parser.add_argument(
        "--dataset_base",
        default=None,
    )

    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    # parser.add_argument(
    #     "-f",
    #     "--postfix",
    #     type=str,
    #     default="",
    #     help="post-postfix for default name",
    # )
    parser.add_argument(
        "-o",
        "--outckpt",
        type=str,
        default="",
        help="path for output ckpt",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="outputs/",
        help="directory for logging dat shit",
    )
    # parser.add_argument(
    #     "--scale_lr",
    #     type=str2bool,
    #     nargs="?",
    #     const=True,
    #     default=True,
    #     help="scale base-lr by ngpu * batch_size * n_accumulate",
    # )
    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    if "RANK" in os.environ:
        node_rank = int(os.environ.get('RANK'))
    else:
        node_rank = int(os.environ.get('LOCAL_RANK', 0))
    opt, unknown = parser.parse_known_args()

    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))

        opt.resume_from_checkpoint = opt.resume
        logdir = opt.logdir
        # logdir = opt.logdir + opt.postfix
    else:
        logdir = opt.logdir
        # logdir = opt.logdir + opt.postfix
        opt.resume_from_checkpoint = None

    # setup trainer loggers
    os.makedirs(logdir, exist_ok=True)
    pl_logger = setup_logger(output=logdir, distributed_rank=node_rank, name='VTDM')

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
   
    node_seed = int(str(opt.seed) + str(node_rank).zfill(2))
    seed_everything(node_seed)

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    dataset_config = config if opt.dataset_base is None else OmegaConf.load(opt.dataset_base)
    lightning_config = config.pop("lightning", OmegaConf.create())

    # merge trainer cli with config
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    cpu = False

    if not opt.no_test or opt.debug:
        trainer_config["gpus"] = '0,'
    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)
    if not "gpus" in trainer_config:
        del trainer_config["accelerator"]
        cpu = True
    else:
        gpuinfo = trainer_config["gpus"]
        pl_logger.info(f"Running on GPUs {gpuinfo}")
        cpu = False
        os.environ["GPU_PER_NODE"] = str(len(gpuinfo.split(',')))
    if "strategy" in trainer_config:
        trainer_config['accelerator'] = "cuda"
        pl_logger.info("Use the strategy of {}".format(trainer_config['strategy']))
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    pl_logger.info(f"Pytorch lightning trainer config: \n{trainer_config}")

    # data
    data = instantiate_from_config(dataset_config.data)
    data.setup()
    pl_logger.info("Set up dataset.")

    # model init
    model = instantiate_from_config(config.model)

    # trainer and callbacks
    trainer_kwargs = dict()

    # default configs
    default_logger_cfgs = {
        "testtube": {
            "target": "pytorch_lightning.loggers.TestTubeLogger",
            "params": {
                "name": "testtube",
                "save_dir": logdir,
            }
        },
    }
    default_logger_cfg = default_logger_cfgs["testtube"]
    if "logger" in lightning_config:
        logger_cfg = lightning_config.logger
    else:
        logger_cfg = OmegaConf.create()
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

    default_modelckpt_cfg = {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {
            "dirpath": ckptdir,
            "filename": "{epoch:06}",
            "verbose": True,
            'save_weights_only': True
        }
    }
    if hasattr(model, "monitor"):
        pl_logger.info(f"Monitoring {model.monitor} as checkpoint metric.")
        default_modelckpt_cfg["params"]["monitor"] = model.monitor
        default_modelckpt_cfg["params"]["save_top_k"] = 10

    if "modelcheckpoint" in lightning_config:
        modelckpt_cfg = lightning_config.modelcheckpoint
    else:
        modelckpt_cfg = OmegaConf.create()
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
    pl_logger.info(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
    if version.parse(pl.__version__) < version.parse('1.4.0'):
        trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

    # add callback which sets up log directory
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "vtdm.callbacks.SetupCallback",
            "params": {
                "resume": opt.resume,
                "now": now,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": config,
                "lightning_config": lightning_config,
            }
        },
        "learning_rate_logger": {
            "target": "train_ddp_spawn.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
            }
        },
        "cuda_callback": {
            "target": "vtdm.callbacks.CUDACallback"
        },
    }
    if version.parse(pl.__version__) >= version.parse('1.4.0'):
        default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()

    if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
        pl_logger.info(
            'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
        default_metrics_over_trainsteps_ckpt_dict = {
            'metrics_over_trainsteps_checkpoint':
                {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                    'params': {
                        "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                        "filename": "{epoch:06}-{step:09}",
                        "verbose": True,
                        'save_top_k': -1,
                        'every_n_train_steps': 10000,
                        'save_weights_only': True
                    }
                    }
        }
        default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
    if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
        callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
    elif 'ignore_keys_callback' in callbacks_cfg:
        del callbacks_cfg['ignore_keys_callback']

    textbar_callbacks = TextProgressBar(pl_logger, trainer_config['logger_refresh_rate'])
    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
    trainer_kwargs["callbacks"].append(textbar_callbacks)

    if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
        pl_logger.info(f"Merged trainsteps-cfg: \n{callbacks_cfg['metrics_over_trainsteps_checkpoint']}")

    pl_logger.info('Done in building trainer kwargs.')

    if "deep_speed_config" in lightning_config:
        from pytorch_lightning.plugins import DeepSpeedPlugin
        pl_logger.info("Use the deep_speed_config of {}".format(lightning_config.deep_speed_config))
        trainer_kwargs['strategy'] = DeepSpeedPlugin(lightning_config.deep_speed_config)

    trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    trainer.logdir = logdir

    # configure learning rate
    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
    if not cpu:
        ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
    else:
        ngpu = 1
    if 'accumulate_grad_batches' in lightning_config.trainer:
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1
    pl_logger.info(f"accumulate_grad_batches = {accumulate_grad_batches}")
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
    model.learning_rate = base_lr
    pl_logger.info("++++ NOT USING LR SCALING ++++")
    pl_logger.info(f"Setting learning rate to {model.learning_rate:.2e}")

    # allow checkpointing via USR1
    def melk(*args, **kwargs):
        # run all checkpoint hooks
        if trainer.global_rank == 0:
            pl_logger.info("Summoning checkpoint.")
            ckpt_path = os.path.join(ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def divein(*args, **kwargs):
        if trainer.global_rank == 0:
            import pudb;
            pudb.set_trace()

    # if not opt.debug:
    if False:
        signal.signal(signal.SIGUSR1, melk) # <-----save before crush, enable if you want to use it
    signal.signal(signal.SIGUSR2, divein)

    # run
    if opt.train:
        try:
            trainer.fit(model, data, ckpt_path=opt.resume_from_checkpoint)

            if trainer.global_rank == 0 and opt.outckpt != '':
                pl_logger.info("Final checkpoint to " + opt.outckpt)
                torch.save(model.state_dict(), opt.outckpt)

        except Exception as exception:
            # if not opt.debug:
            if False:
                melk() # <-----save before crush, enable if you want to use it
            raise exception
    if not opt.no_test and not trainer.interrupted:
        trainer.test(model, data, ckpt_path=opt.resume_from_checkpoint)
    