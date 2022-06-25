import os

import train.shared
from train.datasets_wrapper import HuggingFaceDataset
import re
from train import models


logger = train.shared.logger
ARGS_SPLIT_TOKEN = "^"

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def prepare_dataset_for_training(datasets_dataset):
    """
    将 huggingface 数据集转换格式，转换为合适 tokenizer 处理的格式
    # 这应该是啥也没做，就是把单输入和多输入的数据集进行了标准化
    Changes an `datasets` dataset into the proper format for
    tokenization."""

    def prepare_example_dict(ex):
        """Returns the values in order corresponding to the data.

        ex:
            'Some text input'
        or in the case of multi-sequence inputs:
            ('The premise', 'the hypothesis',)
        etc.
        """
        values = list(ex.values())
        if len(values) == 1:
            return values[0]
        return tuple(values)

    # 这应该是啥也没做，就是把单输入和多输入的数据集进行了标准化
    text, outputs = zip(*((prepare_example_dict(x[0]), x[1]) for x in datasets_dataset))
    # import pdb
    # pdb.set_trace()
    # text = [clean_str(t) for t in text]  # 与攻击时统一。虽然会导致被攻击模型 acc 下降 1-2 个点。
    return list(text), list(outputs)


def dataset_from_args(args):
    """Returns a tuple of ``HuggingFaceDataset`` for the train and test
    datasets for ``args.dataset``.
    根据 'args.dataset' 返回 'HuggingFaceDataset' 的训练集和测试集的 tuple
    """
    dataset_args = args.dataset.split(ARGS_SPLIT_TOKEN)

    # 获取训练集
    if args.dataset_train_split:
        train_dataset = HuggingFaceDataset(
            *dataset_args, split=args.dataset_train_split
        )
    else:
        try:
            train_dataset = HuggingFaceDataset(
                *dataset_args, split="train"
            )
            args.dataset_train_split = "train"
        except KeyError:
            raise KeyError(f"Error: no `train` split found in `{args.dataset}` dataset")
    train_text, train_labels = prepare_dataset_for_training(train_dataset) #传入 HuggingFaceDataset，返回 train_text 和 train_labels 的 list

    # 获取测试集，测试集就分别尝试 dev/eval/validation/test 有哪个就读哪个。有覆盖，最后这四种 split ，读到的作为测试集。
    if args.dataset_dev_split:
        eval_dataset = HuggingFaceDataset(
            *dataset_args, split=args.dataset_dev_split
        )
    else:
        # try common dev split names
        try:
            eval_dataset = HuggingFaceDataset(
                *dataset_args, split="dev"
            )
            args.dataset_dev_split = "dev"
        except KeyError:
            try:
                eval_dataset = HuggingFaceDataset(
                    *dataset_args, split="eval"
                )
                args.dataset_dev_split = "eval"
            except KeyError:
                try:
                    eval_dataset = HuggingFaceDataset(
                        *dataset_args, split="validation"
                    )
                    args.dataset_dev_split = "validation"
                except KeyError:
                    try:
                        eval_dataset = HuggingFaceDataset(
                            *dataset_args, split="test"
                        )
                        args.dataset_dev_split = "test"
                    except KeyError:
                        raise KeyError(
                            f"Could not find `dev`, `eval`, `validation`, or `test` split in dataset {args.dataset}."
                        )
    eval_text, eval_labels = prepare_dataset_for_training(eval_dataset) # 标准化

    return train_text, train_labels, eval_text, eval_labels

def dataset_from_local(args):
    train_text, train_labels, eval_text, eval_labels = [], [], [], []
    if args.dataset in ['mr' or 'imdb']: # single sentence/document input
        with open(args.train_path, 'r', encoding='utf8') as fin:
            for line in fin.readlines():
                label, string = int(line[0]), line[2:].strip()
                train_text.append(string)
                train_labels.append(label)

        with open(args.eval_path, 'r', encoding='utf8') as fin:
            for line in fin.readlines():
                label, string = int(line[0]), line[2:].strip()
                eval_text.append(string)
                eval_labels.append(label)
    else:
        def read_data(filepath):
            import collections
            """
            Read the premises, hypotheses and labels from some NLI dataset's
            file and return them in a dictionary. The file should be in the same
            form as SNLI's .txt files.

            Args:
                filepath: The path to a file containing some premises, hypotheses
                    and labels that must be read. The file should be formatted in
                    the same way as the SNLI (and MultiNLI) dataset.

            Returns:
                A dictionary containing three lists, one for the premises, one for
                the hypotheses, and one for the labels in the input data.
            """
            label2id = {'entailment': 1,
                        'neutral': 2,
                        'contradiction': 0}  # 与 hugging face 要一致 # TODD: 检验 现在这个映射应该不对， 也不是 0，1，2，直接debug看
            import re
            _RE_COMBINE_WHITESPACE = re.compile(r"\s+")

            with open(filepath, 'r', encoding='utf8') as input_data:
                inputs, labels = [], []

                # Translation tables to remove parentheses and punctuation from
                # strings.
                parentheses_table = str.maketrans({'(': None, ')': None})

                # Ignore the headers on the first line of the file.
                next(input_data)

                for line in input_data:
                    line = line.strip().split('\t')

                    # Ignore sentences that have no gold label.
                    if line[0] == '-':
                        continue

                    premise = line[1]
                    hypothesis = line[2]

                    # Remove '(' and ')' from the premises and hypotheses.
                    premise = premise.translate(parentheses_table)
                    hypothesis = hypothesis.translate(parentheses_table)

                    # Substitute multiple space to one
                    premise = _RE_COMBINE_WHITESPACE.sub(" ", premise).strip()
                    hypothesis = _RE_COMBINE_WHITESPACE.sub(" ", hypothesis).strip()

                    # input = collections.OrderedDict(
                    #     [('premise', premise),
                    #      ('hypothesis', hypothesis)]
                    # )
                    input = (premise, hypothesis)
                    inputs.append(input)

                    labels.append(label2id[line[0]])

                return inputs, labels

        train_text, train_labels = read_data(args.train_path)
        eval_text, eval_labels = read_data(args.eval_path)

    return train_text, train_labels, eval_text, eval_labels




def model_from_args(train_args, num_labels, model_path=None):
    """Constructs a model from its `train_args.json`.
    根据参数创建模型。（只有 run_training 用到了这个函数，那 attacks 的时候载入模型用啥函数啊？）（都用了，attacks 也这样读取 victim 模型
    若是 TextAttack 的 lstm/cnn 模型，则通过 textattack.models 进行创建
    若是 huggingface 的 bert-based 模型，则传入模型名称，从 huggingface model hub 上载入
    如果设置了 model_path ，则从本地载入权重

    If huggingface model, loads from model hub address. If TextAttack
    lstm/cnn, loads from disk (and `model_path` provides the path to the
    model).
    """
    if train_args.model == "lstm":
        # 传入模型类型为 lstm
        train.shared.logger.info("Loading model: LSTMForClassification")
        model = models.helpers.LSTMForClassification(
            max_seq_length=train_args.max_length, # 原来 max_length 是在模型里设置，而不是在 dataloader 里设置的
            num_labels=num_labels, # 类别数量
            emb_layer_trainable=False, # embedding 不微调
        )
        if model_path:
            model.load_from_disk(model_path)

        model = models.wrappers.PyTorchModelWrapper(model, model.tokenizer) # 这 tokenizer 是继承自 nn.Model 的
    elif train_args.model == "cnn":
        # cnn 同理
        train.shared.logger.info(
            "Loading model: WordCNNForClassification"
        )
        model = models.helpers.WordCNNForClassification(
            max_seq_length=train_args.max_length,
            num_labels=num_labels,
            emb_layer_trainable=False,
        )
        if model_path:
            model.load_from_disk(model_path)

        model = models.wrappers.PyTorchModelWrapper(model, model.tokenizer)  # 把 tokenizer 和 model 打包在一起
    else:
        # 如果是 huggingface 的模型
        import transformers

        train.shared.logger.info(
            f"Loading transformers AutoModelForSequenceClassification: {train_args.model}"
        )
        # 创建 AutoConfig
        # import pdb
        # pdb.set_trace()
        if train_args.mixup_training and train_args.adversarial_training:
            model = models.helpers.MixBert(train_args.model, num_labels=num_labels, finetuning_task=train_args.dataset )
        elif 'vibert' in train_args.model_type:
            config = transformers.AutoConfig.from_pretrained(
                train_args.model, num_labels=num_labels,
                finetuning_task=train_args.dataset,
                output_hidden_states=True
            )
            ib_dim = 144
            hidden_dim = (768 + ib_dim) // 2
            # sets the parameters of IB or MLP baseline.
            config.ib = True
            config.activation = "relu"
            config.hidden_dim = hidden_dim
            config.ib_dim = ib_dim
            config.beta = train_args.beta  #1e-05
            config.sample_size = 5
            config.kl_annealing = "linear"
            config.deterministic = None
            config.model_cache = train_args.model
            # import pdb
            # pdb.set_trace()
            model = models.helpers.ViBertForClassification(
                config=config,
            )
        elif 'disen_vi' in train_args.model_type:
            from train.methods.disen_vi_uda.config import ConfigTrain
            from train.shared_libs.utils.auto_tokenizer import AutoTokenizer
            from train.methods.disen_vi_uda.disen_vi import Disen_VI
            load_rel_path = train_args.load_rel_path
            cfg = ConfigTrain(load_rel_path=load_rel_path)
            tokenizer = AutoTokenizer(train_args.model, use_fast=True, max_length=cfg.args.max_len)
            cfg.args.vocab_size = tokenizer.tokenizer.vocab_size
            cfg.args.model_cache = train_args.model
            # 2. Generate model & dataloader
            # dataloader = generate_data(cfg)
            train_args.max_length = cfg.args.max_len
            model = Disen_VI(cfg=cfg)
            model._load_checkpoint()
        elif 'disenIB_adv' in train_args.model_type:
            from train.methods.disenib_adv.config import ConfigTrain
            from train.shared_libs.utils.auto_tokenizer import AutoTokenizer
            from train.methods.disenib_adv.disenib_adv import DisenIB_Adv
            load_rel_path = train_args.load_rel_path
            cfg = ConfigTrain(load_rel_path=load_rel_path)
            tokenizer = AutoTokenizer(train_args.model, use_fast=True, max_length=cfg.args.max_len)
            cfg.args.vocab_size = tokenizer.tokenizer.vocab_size
            cfg.args.model_cache = train_args.model
            # 2. Generate model & dataloader
            # dataloader = generate_data(cfg)
            train_args.max_length = cfg.args.max_len
            model = DisenIB_Adv(cfg=cfg)
            model._load_checkpoint()
        elif 'disenIB_mix' in train_args.model_type:
            from train.methods.disenib_mix.config import ConfigTrain
            from train.shared_libs.utils.auto_tokenizer import AutoTokenizer
            from train.methods.disenib_mix.disenib_mix import DisenIB_Mix
            load_rel_path = train_args.load_rel_path
            cfg = ConfigTrain(load_rel_path=load_rel_path)
            tokenizer = AutoTokenizer(train_args.model, use_fast=True, max_length=cfg.args.max_len)
            cfg.args.vocab_size = tokenizer.tokenizer.vocab_size
            cfg.args.model_cache = train_args.model
            # 2. Generate model & dataloader
            # dataloader = generate_data(cfg)
            train_args.max_length = cfg.args.max_len
            model = DisenIB_Mix(cfg=cfg)
            model._load_checkpoint()
        elif 'disenIB' in train_args.model_type:
            from train.methods.disenib_fc.config import ConfigTrain
            from train.shared_libs.utils.auto_tokenizer import AutoTokenizer
            from train.methods.disenib_fc.disenib import DisenIB
            load_rel_path = train_args.load_rel_path
            cfg = ConfigTrain(load_rel_path=load_rel_path)
            tokenizer = AutoTokenizer(train_args.model, use_fast=True, max_length=cfg.args.max_len)
            cfg.args.vocab_size = tokenizer.tokenizer.vocab_size
            cfg.args.model_cache = train_args.model
            # 2. Generate model & dataloader
            # dataloader = generate_data(cfg)
            train_args.max_length = cfg.args.max_len
            model = DisenIB(cfg=cfg)
            model._load_checkpoint()
        elif 'infobert' == train_args.model_type:
            from .textdefender_args import ProgramArgs
            from .textdefender import AttackBenchmarkTask
            args = ProgramArgs.parse()
            args.training_type = train_args.model_type #'infobert'
            # 数据集
            args.dataset = train_args.dataset #'mr'
            if args.dataset == 'snli':
                args.epochs = train_args.epochs
                args.dataset_name = train_args.dataset
            args.mode = 'attacks'
            args.model_name_or_path = train_args.model
            from train.shared.utils import device
            args.device = device
            train_args.max_length = args.max_seq_len
            model_wrapper = AttackBenchmarkTask(args)
            model = model_wrapper.get_model(args)
            # tokenizer = model_wrapper.tokenizer
        elif 'ascc' == train_args.model_type:
            from .textdefender_args import ProgramArgs
            from .textdefender import AttackBenchmarkTask
            args = ProgramArgs.parse()
            args.training_type = train_args.model_type #'infobert'
            # 数据集
            args.dataset = train_args.dataset #'mr'
            if args.dataset == 'snli':
                args.epochs = train_args.epochs
                args.dataset_name = train_args.dataset
            args.mode = 'attacks'
            args.model_name_or_path = train_args.model
            args.exp = train_args.exp
            args.alpha = 10.0
            args.beta = 4.0
            args.num_steps = 5
            args.nbr_file = '/root/zengdajun_2_1/data/zjh/DisenADA/train/data/external/euc-top8-d0.7.json'
            from train.shared.utils import device
            args.device = device
            train_args.max_length = args.max_seq_len
            model_wrapper = AttackBenchmarkTask(args)
            model = model_wrapper.get_model(args)
        elif 'tavat' == train_args.model_type:
            from .textdefender_args import ProgramArgs
            from .textdefender import AttackBenchmarkTask
            args = ProgramArgs.parse()
            args.training_type = train_args.model_type #'infobert'
            # 数据集
            args.dataset = train_args.dataset #'mr'
            if args.dataset == 'snli':
                args.epochs = train_args.epochs
                args.dataset_name = train_args.dataset
            args.mode = 'attacks'
            args.model_name_or_path = train_args.model
            args.use_global_embedding = True
            from train.shared.utils import device
            args.device = device
            train_args.max_length = args.max_seq_len
            model_wrapper = AttackBenchmarkTask(args)
            model = model_wrapper.get_model(args)
            # tokenizer = model_wrapper.tokenizer
        else:
            config = transformers.AutoConfig.from_pretrained(
                train_args.model, num_labels=num_labels, finetuning_task=train_args.dataset  # finetune 过的数据集
            )
            model = transformers.AutoModelForSequenceClassification.from_pretrained(  # 序列分类模型
                train_args.model,
                config=config,
            )
        tokenizer = models.tokenizers.AutoTokenizer(
            train_args.model, use_fast=True, max_length=train_args.max_length  # 这里 max_length 是给 tokenizer 的，
        )
        # if model_path and train_args.dataset == 'snli'

        model = models.wrappers.HuggingFaceModelWrapper(model, tokenizer)  # 把 tokenizer 和 model 打包在一起
        if model_path:
            model.load_from_disk(model_path)
    return model


def write_readme(args, best_eval_score, best_eval_score_epoch):
    # Save args to file
    # 保存参数到文件
    readme_save_path = os.path.join(args.output_dir, "README.md") # 文件名
    dataset_name = (
        args.dataset.split(ARGS_SPLIT_TOKEN)[0]
        if ARGS_SPLIT_TOKEN in args.dataset
        else args.dataset
    )  # 获取 split 或 dataset，这是不是有问题，难道不应该获取 split 和 dataset 吗
    task_name = "classification" # 任务类型，标签为整数就是分类
    loss_func = "mean squared error" # 损失函数，分类是交叉熵
    metric_name = "accuracy" # 评价指标，分类是准确率
    epoch_info = f"{best_eval_score_epoch} epoch" + ( # 最优模型的轮数
        "s" if best_eval_score_epoch > 1 else ""
    )
    # 文本内容
    readme_text = f"""
## Model Card

This `{args.model}` model was trained or fine-tuned for sequence classification 
and the {dataset_name} dataset loaded using the `datasets` library. The model was trained of fine-tuned
for {args.num_train_epochs} epochs with a batch size of {args.batch_size}, a learning
rate of {args.learning_rate}, and a maximum sequence length of {args.max_length}.
Since this was a {task_name} task, the model was trained with a {loss_func} loss function.
The best score the model achieved on this task was {best_eval_score}, as measured by the
eval set {metric_name}, found after {epoch_info}.

"""

    with open(readme_save_path, "w", encoding="utf-8") as f:
        f.write(readme_text.strip() + "\n")
    logger.info(f"Wrote README to {readme_save_path}.")
