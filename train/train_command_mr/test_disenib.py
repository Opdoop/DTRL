import sys
from args_helper import _path
sys.path.extend(_path())


from train.methods.disenib_fc.config import ConfigTrain
from train.shared_libs.utils.auto_tokenizer import AutoTokenizer
from train.methods.disenib_fc.disenib import DisenIB
import torch
import numpy as np

if __name__ == '__main__':
    # 1. Generate config
    load_rel_path = 'unspecified/RandSeed[3601]/params/config[599].pkl'
    cfg = ConfigTrain(load_rel_path=load_rel_path)
    tokenizer = AutoTokenizer('../../../bert-base-uncased', use_fast=True, max_length=cfg.args.max_len)
    cfg.args.vocab_size = tokenizer.tokenizer.vocab_size
    # 2. Generate model & dataloader
    # dataloader = generate_data(cfg)
    model = DisenIB(cfg=cfg)
    # 3. Train

    model._load_checkpoint()
    text = 'Some text for test forward process'
    input = tokenizer.encode(text)
    input = {
        k: torch.tensor(np.array(v)) for k, v in input.items()
    }
    output = model(input)
