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
    args = ProgramArgs.parse(True)
    # 根据具体实验设置相应参数
    # 模型类型
    args.training_type = 'infobert'
    # 数据集
    args.dataset = 'mr'
    args.train_path = '../data/mr/train.txt'   # dataloader 方法修改
    args.eval_path = '../data/mr/test.txt'

    args.build_environment()
    args.build_logging()
    logging.info(args)
    test = AttackBenchmarkTask(args)

    test.methods[args.mode](args)