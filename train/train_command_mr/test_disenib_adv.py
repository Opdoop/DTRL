import sys
from args_helper import _path
sys.path.extend(_path())

from visulization.bert_cls_representation.util import make_dataloader
from train.methods.disenib_adv.config import ConfigTrain
from train.shared_libs.utils.auto_tokenizer import AutoTokenizer
from train.methods.disenib_adv.disenib_adv import DisenIB_Adv
import torch
import numpy as np

if __name__ == '__main__':
    # 1. Generate config
    load_rel_path = 'unspecified/RandSeed[3101]/params/config[599].pkl'
    cfg = ConfigTrain(load_rel_path=load_rel_path)
    tokenizer = AutoTokenizer('../../../bert-base-uncased', use_fast=True, max_length=cfg.args.max_len)
    cfg.args.vocab_size = tokenizer.tokenizer.vocab_size
    # 2. Generate model & dataloader
    # dataloader = generate_data(cfg)
    model = DisenIB_Adv(cfg=cfg)
    # 3. Train
    text, labels = [], []
    file_path = '/root/data/zjh/MixAdv/attacks/data/mr'
    with open(file_path, 'r', encoding='utf8') as fin:
        for line in fin.readlines():
            label, string = int(line[0]), line[2:].strip()
            text.append(string)
            labels.append(label)

    dataloader = make_dataloader(tokenizer, text, labels, batch_size=4)
    model._load_checkpoint()
    model.to('cuda')
    text = 'Some text for test forward process'
    pred, gt = [], []
    for batch in dataloader:
        input, ground_label = batch
        input = {
            k: torch.stack(v).T.to('cuda') for k, v in input.items()
        }
        output = model(input)
        cur_pred = torch.argmax(output['logits'], dim=1)
        pred.append(cur_pred.detach().cpu().numpy())
        gt.append(ground_label.detach().cpu().numpy())
    acc = 0
    pred, gt = np.concatenate(pred), np.concatenate(gt)
    for i in range(len(pred)):
        import pdb
        pdb.set_trace()
        if pred[i] == gt[i]:
            acc += 1
    print(acc)