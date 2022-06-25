import sys
from args_helper import _path
sys.path.extend(_path())

from train.methods.disenib_mix.config import ConfigTrain
from train.methods.disenib_mix.dataloader import generate_data
from train.methods.disenib_mix.disenib_mix import DisenIB_Mix


if __name__ == '__main__':
    # 1. Generate config
    cfg = ConfigTrain()
    # 2. Generate model & dataloader
    dataloader = generate_data(cfg)
    model = DisenIB_Mix(cfg=cfg)
    # 3. Train
    # import pdb
    # pdb.set_trace()
    model.train_parameters(**dataloader)
