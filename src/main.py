import argparse

from utils import create_logger
import torch
import numpy as np
import os
import time
from train_model import Trainer_KBQA
from parsing import add_parse_args
import random
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['http_proxy']="http://127.0.0.1:7890"
os.environ['https_proxy']="http://127.0.0.1:7890"
parser = argparse.ArgumentParser()
add_parse_args(parser)

args = parser.parse_args()
args.use_cuda = torch.cuda.is_available()


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# 如果没提供实验名称会自动生成名称，如Experiment name: mnist-lenet-1682075363
if args.experiment_name == None:
    timestamp = str(int(time.time()))
    args.experiment_name = "{}-{}-{}".format(
        args.dataset,
        args.model_name,
        timestamp,
    )


def main():
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    logger = create_logger(args)
    
    trainer = Trainer_KBQA(args=vars(args), model_name=args.model_name, logger=logger)

    if not args.is_eval:
        # 训练模式
        trainer.train(0, args.num_epoch - 1)
    else:
        assert args.load_experiment is not None
        if args.load_experiment is not None:
            ckpt_path = os.path.join(args.checkpoint_dir, args.load_experiment)
            print("Loading pre trained model from {}".format(ckpt_path))
        else:
            ckpt_path = None
        #评估模式
        trainer.evaluate_single(ckpt_path)


if __name__ == '__main__':
    main()
