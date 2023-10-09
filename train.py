import argparse
import random
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.train.model import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore import dataset as ds
from mindspore import load_checkpoint, load_param_into_net

from utils.gpu import set_gpu
from utils.parse import parse_yaml

import os
import time
os.environ["TZ"] = "UTC-8"
time.tzset()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BI-RADS Classification')
    parser.add_argument('--isFoldValue', type=bool, default=False,
                        help='using cross-validation')
    parser.add_argument('--num_folds', type=int, default=5,
                        help='num of folds')
    parser.add_argument('--seed', type=int, default=22,
                        help='random seed for training. default=22')
    parser.add_argument('--use_cuda', default='true', type=str,
                        help='whether use cuda. default: true')
    parser.add_argument('--use_parallel', default='false', type=str,
                        help='whether use cuda. default: false')
    parser.add_argument('--gpu', default='all', type=str,
                        help='use gpu device. default: all')
    parser.add_argument('--config', default='cfgs/default.yaml', type=str,
                        help='configuration file. default=cfgs/default.yaml')
    parser.add_argument('--model', default='sil_model', type=str,
                        help='choose model. default=sil_model')
    parser.add_argument('--description', default='None', type=str,
                        help='description of experiment')
    parser.add_argument('--net', default='inception_v3', type=str,
                        help='choose net. default=inception_v3')
    parser.add_argument('--infer', default='false', type=str,
                        help='do inference. default: false')
    parser.add_argument('--ckpt_path', default=None, type=str)

    args, _ = parser.parse_known_args()
    num_gpus = set_gpu(args.gpu)

    np.random.seed(args.seed)
    random.seed(args.seed)
    context.set_seed(args.seed)

    config = parse_yaml(args.config)

    config['data']['is_fold'] = False
    config['data']['num_folds'] = args.num_folds

    config['eval']['ckpt_path'] = args.ckpt_path

    network_params = config['network']
    network_params['seed'] = args.seed
    network_params['device'] = "cuda" if str2bool(args.use_cuda) else "cpu"
    network_params['use_parallel'] = str2bool(args.use_parallel)
    do_inference = str2bool(args.infer)
    network_params['num_gpus'] = num_gpus
    network_params['model_name'] = args.net
    network_params['description'] = args.description

    if args.model == 'mv_model':
        from models.mv_model import Model
    else:
        raise NotImplementedError(args.model+" not implemented")

    model = Model(config)

    if args.ckpt_path:
        param_dict = load_checkpoint(args.ckpt_path)
        load_param_into_net(model, param_dict)

    model.run()