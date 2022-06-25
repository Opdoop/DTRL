import sys
from args_helper import _path
sys.path.extend(_path())

from train.methods.disenib_fc.config import ConfigTrain
from train.methods.disenib_fc.dataloader import generate_data
from train.methods.disenib_fc.disenib import DisenIB


if __name__ == '__main__':
    # 1. Generate config
    cfg = ConfigTrain()
    # 2. Generate model & dataloader
    cfg.dataset = 'snli'
    cfg.learning_rate = 2e-5
    dataloader = generate_data(cfg)
    model = DisenIB(cfg=cfg)
    # 3. Train
    # import pdb
    # pdb.set_trace()
    model.train_parameters(**dataloader)
