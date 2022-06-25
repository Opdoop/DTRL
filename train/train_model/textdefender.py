import torch
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from .textdefender_args import ProgramArgs, string_to_bool

import logging
from typing import Union

from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import AdamW
import numpy as np

from datasets_wrapper.reader import ClassificationReader
from trainer import (
    BaseTrainer,
    FreeLBTrainer,
    HotflipTrainer,
    PGDTrainer,
    IBPTrainer,
    TokenAwareVirtualAdversarialTrainer,
    InfoBertTrainer,
    MixUpTrainer,
    ASCCTrainer
)
from utils.config import MODEL_CLASSES, DATASET_LABEL_NUM
from utils.metrics import Metric, ClassificationMetric
from utils.my_utils import convert_batch_to_bert_input_dict
from utils.public import auto_create, check_and_create_path
from utils.dne_utils import get_bert_vocab
from utils.ascc_utils import WarmupMultiStepLR

from models.helpers.textdefender_models import MixText, ASCCModel


class AttackBenchmarkTask(object):
    def __init__(self, args: ProgramArgs):
        self.methods = {'train': self.train,
                        'evaluate': self.evaluate,
                        'attacks': self.attack,
                        }
        assert args.mode in self.methods, 'mode {} not found'.format(args.mode)
        if args.mode in ['train', 'evaluate']:
            self.tensor_input = False if args.training_type in args.type_accept_instance_as_input and args.mode == 'train' else True

            self.tokenizer = self._build_tokenizer(args)
            self.dataset_reader = ClassificationReader(model_type=args.model_type, max_seq_len=args.max_seq_len)

            # self.train_raw, self.eval_raw, self.test_raw = auto_create(
            #     f'{args.dataset_name}_raw_datasets', lambda: self._build_raw_dataset(args),
            #     True, path=args.cache_path
            # )
            self.train_raw, self.eval_raw, self.test_raw = self._build_raw_dataset(args)
            self.train_dataset, self.eval_dataset, self.test_dataset = self._build_tokenized_dataset(args)
            if not self.tensor_input:
                self.train_dataset = self.train_raw

            self.data_loader, self.eval_data_loader, self.test_data_loader = self._build_dataloader(args)
            self.loss_function = self._build_criterion(args)
        self.model = self._build_model(args)

    def train(self, args: ProgramArgs):
        self.optimizer = self._build_optimizer(args)
        self.lr_scheduler = self._build_lr_scheduler(args)
        self.writer = self._build_writer(args)
        trainer = self._build_trainer(args)
        best_metric = None
        epoch_now = self._check_training_epoch(args)
        for epoch_time in range(epoch_now, args.epochs):
            if args.training_type == 'ibp':
                trainer.set_epoch(epoch_time)
            trainer.train_epoch(args, epoch_time)

            # saving model according to epoch_time
            self._saving_model_by_epoch(args, epoch_time)

            # evaluate model according to epoch_time
            metric = self.evaluate(args, is_training=True)

            # update best metric
            # if best_metric is None, update it with epoch metric directly, otherwise compare it with epoch_metric
            if best_metric is None or metric > best_metric:
                best_metric = metric
                self._save_model_to_file(f"{args.workspace}/outputs/training/{args.dataset_name}-{args.training_type}-{args.model_type}-{args.exp}",
                                         args.build_saving_file_name(description='best'))

        # if args.saving_last_epoch:
        #     print('Saving last epoch')
        #     self._save_model_to_file(f"{args.workspace}/outputs/training/{args.dataset_name}-{args.training_type}-{args.model_type}",
        #                              args.build_saving_file_name(description='best'))

        self.evaluate(args)

    @torch.no_grad()
    def evaluate(self, args: ProgramArgs, is_training: bool = False) -> Metric:
        if is_training:
            logging.info('Using current modeling parameter to evaluate')
            epoch_iterator = tqdm(self.eval_data_loader)
        else:
            self._loading_model_from_file(f"{args.workspace}/outputs/training/{args.dataset_name}-{args.training_type}-{args.model_type}-{args.exp}",
                                          args.build_saving_file_name(description='best'))
            epoch_iterator = tqdm(self.test_data_loader)
        self.model.eval()

        metric = ClassificationMetric(compare_key=args.compare_key)
        for step, batch in enumerate(epoch_iterator):
            assert isinstance(batch[0], torch.Tensor)
            batch = tuple(t.to(args.device) for t in batch)
            golds = batch[3]
            inputs = convert_batch_to_bert_input_dict(batch, args.model_type)
            logits = self.model.forward(**inputs)[0]
            losses = self.loss_function(logits.view(-1, DATASET_LABEL_NUM[args.dataset_name]), golds.view(-1))
            epoch_iterator.set_description('loss: {:.4f}'.format(torch.mean(losses)))
            metric(losses, logits, golds)

        print(metric)
        logging.info(metric)
        return metric

    def attack(self, args: ProgramArgs):
        if args.use_dev_aug == 'False':
            self._loading_model_from_file(f"{args.workspace}/outputs/training/{args.dataset_name}-{args.training_type}-{args.model_type}-{args.exp}",
                                      args.build_saving_file_name(description='best'))
        else:
            self._loading_model_from_file(f"{args.workspace}/outputs/training/{args.dataset_name}-{args.training_type}-{args.model_type}-{args.exp}",
                                      args.build_saving_file_name(description=f"best_{args.dev_aug_attacker}_{args.dev_aug_ratio}"))
        self.model.eval()
        attacker = self._build_attacker(args)

        if args.evaluation_data_type == 'dev':
            dataset = self.eval_raw
        else:
            dataset = self.test_raw
        test_instances = dataset

        attacker_log_path = '{}'.format(args.build_logging_path())
        attacker_log_path = os.path.join(f"{args.workspace}/log/{args.dataset_name}_{args.training_type}",
                                         attacker_log_path)
        attacker_log_manager = None #AttackLogManager()
        # attacker_log_manager.enable_stdout()
        attacker_log_manager.add_output_file(
            os.path.join(attacker_log_path, f'{args.attack_method}_{args.neighbour_vocab_size}_{args.modify_ratio}.txt'))
        test_instances = [x for x in test_instances if len(x.text_a.split(' ')) > 4]
        # attacks multiple times for average success rate
        for i in range(args.attack_times):
            print("Attack time {}".format(i))
            choice_instances = np.random.choice(test_instances, size=(args.attack_numbers,), replace=False)
            dataset = None #CustomTextAttackDataset.from_instances(args.dataset_name, choice_instances, self.dataset_reader.get_labels())
            results_iterable = attacker.attack_dataset(dataset)
            description = tqdm(results_iterable, total=len(choice_instances))
            result_statistics = None # SimplifidResult()
            for result in description:
                try:
                    attacker_log_manager.log_result(result)
                    result_statistics(result)
                    description.set_description(result_statistics.__str__())
                except Exception as e:
                    print('error in process')
                    continue

        attacker_log_manager.enable_stdout()
        attacker_log_manager.log_summary()

    def _save_model_to_file(self, save_dir: str, file_name: str):
        save_file_name = '{}.pth'.format(file_name)
        check_and_create_path(save_dir)
        save_path = os.path.join(save_dir, save_file_name)
        torch.save(self.model.state_dict(), save_path)
        logging.info('Saving model to {}'.format(save_path))

    def _saving_model_by_epoch(self, args: ProgramArgs, epoch: int):
        # saving
        if args.saving_step is not None and args.saving_step != 0:
            if (epoch - 1) % args.saving_step == 0:
                self._save_model_to_file(f"{args.workspace}/outputs/training/{args.dataset_name}-{args.training_type}-{args.model_type}-{args.exp}",
                                         args.build_saving_file_name(description='epoch{}'.format(epoch)))

    def _check_training_epoch(self, args: ProgramArgs):
        epoch_now = 0
        save_dir = f"{args.workspace}/outputs/training/{args.dataset_name}-{args.training_type}-{args.model_type}-{args.exp}"
        for epoch in range(args.epochs):
            file_name = args.build_saving_file_name(description='epoch{}'.format(epoch))
            save_file_name = '{}.pth'.format(file_name)
            check_and_create_path(save_dir)
            save_path = os.path.join(save_dir, save_file_name)
            if os.path.exists(save_path) and os.path.isfile(save_path):
                epoch_now = epoch + 1
                continue
            else:
                if epoch_now != 0:
                    file_name = args.build_saving_file_name(description='epoch{}'.format(epoch-1))
                    self._loading_model_from_file(save_dir, file_name)
                break
        return epoch_now

    def _loading_model_from_file(self, save_dir: str, file_name: str):
        load_file_name = '{}.pth'.format(file_name)
        load_path = os.path.join(save_dir, load_file_name)
        assert os.path.exists(load_path) and os.path.isfile(load_path), '{} not exits'.format(load_path)
        self.model.load_state_dict(torch.load(load_path), strict=False)
        logging.info('Loading model from {}'.format(load_path))

    def _build_trainer(self, args: ProgramArgs):
        trainer = BaseTrainer(self.data_loader, self.model, self.loss_function, self.optimizer, self.lr_scheduler,
                              self.writer)
        if args.training_type == 'freelb':
            trainer = FreeLBTrainer(self.data_loader, self.model, self.loss_function, self.optimizer, self.lr_scheduler,
                                    self.writer)
        elif args.training_type == 'pgd':
            trainer = PGDTrainer(self.data_loader, self.model, self.loss_function, self.optimizer, self.lr_scheduler,
                                 self.writer)
        elif args.training_type == 'advhotflip':
            trainer = HotflipTrainer(args, self.tokenizer, self.data_loader, self.model, self.loss_function,
                                     self.optimizer, self.lr_scheduler, self.writer)
        elif args.training_type == 'ibp':
            trainer = IBPTrainer(args, self.data_loader, self.model, self.loss_function, self.optimizer,
                                 self.lr_scheduler, self.writer)
        elif args.training_type == 'tavat':
            trainer = TokenAwareVirtualAdversarialTrainer(args, self.data_loader, self.model, self.loss_function,
                                                          self.optimizer, self.lr_scheduler, self.writer)
        elif args.training_type == 'infobert':
            trainer = InfoBertTrainer(args, self.data_loader, self.model, self.loss_function,
                                      self.optimizer, self.lr_scheduler, self.writer)
        elif args.training_type == 'mixup':
            trainer = MixUpTrainer(args, self.data_loader, self.model, self.loss_function,
                                   self.optimizer, self.lr_scheduler, self.writer)
        elif args.training_type == 'ascc':
            trainer = ASCCTrainer(args, self.data_loader, self.model, self.loss_function,
                                 self.optimizer, self.lr_scheduler, self.writer)

        return trainer

    def _build_optimizer(self, args: ProgramArgs, **kwargs):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        return optimizer

    def _build_model(self, args: ProgramArgs):
        if args.training_type == 'mixup':
            config_class, _, _ = MODEL_CLASSES[args.model_type]
            config = config_class.from_pretrained(
                args.model_name_or_path,
                num_labels=DATASET_LABEL_NUM[args.dataset_name],
                finetuning_task=args.dataset_name,
                output_hidden_states=True,
            )
            model = MixText.from_pretrained(
                args.model_name_or_path,
                from_tf=bool('ckpt' in args.model_name_or_path),
                config=config
            ).to(args.device)
        elif args.training_type == 'ascc':
            config_class, _, _ = MODEL_CLASSES[args.model_type]
            config = config_class.from_pretrained(
                args.model_name_or_path,
                num_labels=DATASET_LABEL_NUM[args.dataset_name],
                finetuning_task=args.dataset_name,
                output_hidden_states=True,
            )
            model = ASCCModel.from_pretrained(
                args.model_name_or_path,
                from_tf=bool('ckpt' in args.model_name_or_path),
                config=config
            ).to(args.device)
            bert_vocab = get_bert_vocab(args.model_name_or_path)
            model.build_nbrs(args.nbr_file, bert_vocab, args.alpha, args.num_steps)
        else:
            config_class, model_class, _ = MODEL_CLASSES[args.model_type]
            config = config_class.from_pretrained(
                args.model_name_or_path,
                num_labels=DATASET_LABEL_NUM[args.dataset_name],
                finetuning_task=args.dataset_name,
                output_hidden_states=True,
            )
            model = model_class.from_pretrained(
                args.model_name_or_path,
                from_tf=bool('ckpt' in args.model_name_or_path),
                config=config
            ).to(args.device)
        return model

    def _build_tokenizer(self, args: ProgramArgs):
        _, _, tokenizer_class = MODEL_CLASSES[args.model_type]
        tokenizer = tokenizer_class.from_pretrained(
            args.model_name_or_path,
            do_lower_case=string_to_bool(args.do_lower_case)
        )
        return tokenizer

    def _build_raw_dataset(self, args: ProgramArgs):
        '''
        先 build_raw, 然后 _build_tokenized_dataset 会调用 self.train_raw
        :param args:
        :return:
        '''
        train_raw, eval_raw = self.dataset_reader.read_from_file(args)
        test_raw = eval_raw
        return train_raw, eval_raw, test_raw

    def _build_tokenized_dataset(self, args: ProgramArgs):
        assert isinstance(self.dataset_reader, ClassificationReader)
        train_dataset = self.dataset_reader.get_dataset(self.train_raw, self.tokenizer)
        eval_dataset = self.dataset_reader.get_dataset(self.eval_raw, self.tokenizer)
        test_dataset = self.dataset_reader.get_dataset(self.test_raw, self.tokenizer)

        return train_dataset, eval_dataset, test_dataset

    def _build_dataloader(self, args: ProgramArgs):
        assert isinstance(self.dataset_reader, ClassificationReader)
        train_data_loader = self.dataset_reader.get_dataset_loader(dataset=self.train_dataset,
                                                                       tokenized=self.tensor_input,
                                                                       batch_size=args.batch_size,
                                                                       shuffle=string_to_bool(args.shuffle))
        eval_data_loader = self.dataset_reader.get_dataset_loader(dataset=self.eval_dataset,
                                                                  tokenized=True,
                                                                  batch_size=args.batch_size,
                                                                  shuffle=string_to_bool(args.shuffle))
        test_data_loader = self.dataset_reader.get_dataset_loader(dataset=self.test_dataset,
                                                                  tokenized=True,
                                                                  batch_size=args.batch_size,
                                                                  shuffle=string_to_bool(args.shuffle))
        return train_data_loader, eval_data_loader, test_data_loader

    def _build_criterion(self, args: ProgramArgs):
        return CrossEntropyLoss(reduction='none')

    def _build_lr_scheduler(self, args: ProgramArgs):
        if args.training_type == 'ascc':
            return WarmupMultiStepLR(self.optimizer, (40, 80), 0.1, 1.0 / 10.0, 2, 'linear')
        return CosineAnnealingLR(self.optimizer, len(self.train_dataset) // args.batch_size * args.epochs)

    def _build_writer(self, args: ProgramArgs, **kwargs) -> Union[SummaryWriter, None]:
        writer = None
        if args.tensorboard == 'yes':
            tensorboard_file_name = '{}-tensorboard'.format(args.build_logging_path())
            tensorboard_path = os.path.join(f"{args.workspace}/log/{args.dataset_name}_{args.model_type}",
                                            tensorboard_file_name)
            writer = SummaryWriter(tensorboard_path)
        return writer

    def get_model(self, args: ProgramArgs):
        self._loading_model_from_file(f"{args.workspace}/outputs/training/{args.dataset_name}-{args.training_type}-{args.model_type}-{args.exp}",
                                  args.build_saving_file_name(description='best'))
        self.model.eval()
        print(f"load model from: {args.workspace}/outputs/training/{args.dataset_name}-{args.training_type}-{args.model_type}-{args.exp}",
                                  args.build_saving_file_name(description='best'))
        return self.model

    def _build_attacker(self, args: ProgramArgs):
        # model_wrapper = HuggingFaceModelWrapper(self.model, self.tokenizer, batch_size=args.batch_size) # 这可太香了
        # 从初始化到 _build_attacker 之前打包拿走
        # 之后的逻辑切到自己的 run_train 方法上 > No, 保留，train 和 eval 的部分保留，使用已有的实现，但这需要替换 datloader 部分的方法
        # attacks 测试时候也用这样的逻辑，使用 load custom model, 一直到 model_wrapper
        # problem sovled!

        # attacker = build_english_attacker(args, model_wrapper)
        return None # attacker


if __name__ == '__main__':
    args = ProgramArgs.parse(True)
    args.build_environment()
    args.build_logging()
    logging.info(args)
    test = AttackBenchmarkTask(args)

    test.methods[args.mode](args)
