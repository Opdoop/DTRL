import sys
from args_helper import _path
sys.path.extend(_path())

from train.methods.disenib_adv.config import ConfigTrain
from train.methods.disenib_adv.dataloader import generate_data
from train.methods.disenib_adv.disenib_adv import DisenIB_Adv


if __name__ == '__main__':
    # 1. Generate config
    cfg = ConfigTrain()
    # 2. Generate model & dataloader
    dataloader = generate_data(cfg)
    model = DisenIB_Adv(cfg=cfg)
    # 3. Train
    # import pdb
    # pdb.set_trace()
    model.train_parameters(**dataloader)
