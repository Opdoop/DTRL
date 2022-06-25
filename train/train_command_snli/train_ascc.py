# -*- coding:utf-8 -*-
# Author : Opdoop
# Data : 2022/1/9 21:40

import sys
from args_helper import _path
sys.path.extend(_path())
from train_model.textdefender_args import ProgramArgs
from train_model.textdefender import AttackBenchmarkTask
import logging


if __name__ == "__main__":
    args = ProgramArgs.parse()
    # 根据具体实验设置相应参数
    # 模型类型
    args.training_type = 'ascc'
    # 数据集
    args.dataset_name = 'snli'
    args.dataset = 'snli'
    args.epochs = 5
    args.batch_size = 64
    args.learning_rate = 5e-5
    args.train_path = '../data/snli/snli_1.0_train.txt'
    args.eval_path = '../data/snli/snli_1.0_test.txt'
    args.alpha = 1.0
    args.beta = 3.0
    args.exp = '1-4-5'
    args.num_steps = 5
    args.nbr_file = '/root/zengdajun_2_1/data/zjh/DisenADA/train/data/external/euc-top8-d0.7.json'
    args.build_environment()
    args.build_logging()
    logging.info(args)
    test = AttackBenchmarkTask(args)

    test.methods[args.mode](args)