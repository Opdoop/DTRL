import sys
from args_helper import _path
sys.path.extend(_path())

from train.methods.disen_vi.config import ConfigTrain
from train.methods.disen_vi.dataloader import generate_data
from train.methods.disen_vi.disen_vi import Disen_VI


if __name__ == '__main__':
    # 1. Generate config
    cfg = ConfigTrain()
    # 2. Generate model & dataloader
    dataloader = generate_data(cfg)
    model = Disen_VI(cfg=cfg)
    # 3. Train
    # import pdb
    # pdb.set_trace()
    model.train_parameters(**dataloader)
