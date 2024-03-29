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
    # 通过 TextAttack 训练的模型，只需要提供路径
    # Support loading TextAttack-trained models via just their folder path.
    # If `args.model` is a path/directory, let's assume it was a model
    # trained with textattack, and try and load it.
    current_dir = os.path.dirname(os.path.realpath(__file__))
    model = os.path.realpath(os.path.join(
        current_dir, os.pardir, os.pardir, os.pardir, 'outputs', 'training', model_name
    ))
    model_args_json_path = os.path.join(model, "train_args.json")  # 如果有这个 json 文件
    if not os.path.exists(model_args_json_path):
        raise FileNotFoundError(
            f"Tried to load model from path {model} - could not find train_args.json."
        )
    model_train_args = json.loads(open(model_args_json_path).read())  # 读取参数
    if model_train_args["model"] not in {"cnn", "lstm"}:
        # for huggingface models, set args.model to the path of the model
        model_train_args["model"] = model  # 如果不是 cnn/lstm 模型，赋值为传入的模型文件夹路径
    num_labels = model_train_args["num_labels"]  # 获取分类任务的类别数量

    # 载入模型

    sys.path.append(_path())
    from train.train_model.train_args_helpers import model_from_args
    model = model_from_args(
        argparse.Namespace(**model_train_args),
        num_labels,
        model_path=model,
    )

    return model


model_name = f'bert-base-uncased-snli-mixN-multi'
model_wrapper= load_model_wrapper(model_name)

tokenizer = model_wrapper.tokenizer
model = model_wrapper
# model = model_wrapper.model
# model = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
# import pdb
# pdb.set_trace()