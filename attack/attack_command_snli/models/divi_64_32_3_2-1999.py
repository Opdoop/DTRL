import os
import json
import argparse
import sys

def _path():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    outputs_dir = os.path.realpath(os.path.join(
        current_dir, os.pardir, os.pardir, os.pardir
    ))
    return outputs_dir

def load_model_wrapper(model_name):
    num_labels = 2 #model_train_args["num_labels"]  # 获取分类任务的类别数量
    model_train_args = {'model_type': model_name,
                        'model': '../../../bert-base-uncased',
                        'load_rel_path': 'snli_divi_64_32_3_2/RandSeed[9376]/params/config[1999].pkl',
                        'adversarial_training': None,
                        'mixup_training': None}
    # 载入模型
    sys.path.append(_path())
    from train.train_model.train_args_helpers import model_from_args
    model = model_from_args(
        argparse.Namespace(**model_train_args),
        num_labels,
    )
    return model


model_type = 'disen_vi'
model_wrapper= load_model_wrapper(model_type)

tokenizer = model_wrapper.tokenizer
model = model_wrapper
# 普通的 model
# 沿用 bert 的 tokenizer
# 套在 HuggingFaceModelWrapper 里


# print(model_wrapper.model)
# model = model_wrapper.model
# model = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
# import pdb
# pdb.set_trace()