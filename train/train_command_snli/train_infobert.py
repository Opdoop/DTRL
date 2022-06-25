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
    args.training_type = 'infobert'
    # 数据集
    args.dataset_name = 'snli'
    args.dataset = 'snli'
    args.train_path = '../data/snli/snli_1.0_train.txt'
    args.eval_path = '../data/snli/snli_1.0_test.txt'
    args.epochs = 5

    args.build_environment()
    args.build_logging()
    logging.info(args)
    test = AttackBenchmarkTask(args)

    test.methods[args.mode](args)