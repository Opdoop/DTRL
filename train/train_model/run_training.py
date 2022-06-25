import json
import logging
import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
import tqdm
import transformers
from torch.nn import functional as F

import train.shared

from .perturbed_helper import PerturbedDataset

from .train_args_helpers import (
    dataset_from_args,
    model_from_args,
    write_readme,
    dataset_from_local,
)

device = train.shared.utils.device
logger = train.shared.logger


def _save_args(args, save_path):
    """Dump args dictionary to a json.
    保存传入的参数
    :param: args. Dictionary of arguments to save.
    :save_path: Path to json file to write args to.
    """
    final_args_dict = {k: v for k, v in vars(args).items() if _is_writable_type(v)}
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(final_args_dict, indent=2) + "\n")


def _get_sample_count(*lsts):
    """Get sample count of a dataset.
    获取样本数量，若 X 与 Y 全部对齐，则返回样本数量，否则返回 None
    :param *lsts: variable number of lists.
    :return: sample count of this dataset, if all lists match, else None.
    """
    if all(len(lst) == len(lsts[0]) for lst in lsts):
        sample_count = len(lsts[0])
    else:
        sample_count = None
    return sample_count


def _random_shuffle(*lsts):
    """Randomly shuffle a dataset. Applies the same permutation to each list
    (to preserve mapping between inputs and targets).

    :param *lsts: variable number of lists to shuffle.
    :return: shuffled lsts.
    """
    permutation = np.random.permutation(len(lsts[0]))
    shuffled = []
    for lst in lsts:
        shuffled.append((np.array(lst)[permutation]).tolist())
    return tuple(shuffled)


def _train_val_split(*lsts, split_val=0.2):
    """Split dataset into training and validation sets.
    分割训练集和测试集
    :param *lsts: variable number of lists that make up a dataset (e.g. text, labels)
    :param split_val: float [0., 1.). Fraction of the dataset to reserve for evaluation.
    :return: (train split of list for list in lsts), (val split of list for list in lsts)
    """
    sample_count = _get_sample_count(*lsts)
    if not sample_count:
        raise Exception(
            "Batch Axis inconsistent. All input arrays must have first axis of equal length."
        )
    lsts = _random_shuffle(*lsts)
    split_idx = math.floor(sample_count * split_val)
    train_set = [lst[split_idx:] for lst in lsts]
    val_set = [lst[:split_idx] for lst in lsts]
    if len(train_set) == 1 and len(val_set) == 1:
        train_set = train_set[0]
        val_set = val_set[0]
    return train_set, val_set


def _save_model_checkpoint(model, output_dir, global_step):
    """Save model checkpoint to disk.
    保存模型的 checkpoint
    :param model: 需要保存的模型 Model to save (pytorch)
    :param output_dir: 保存路径 Path to model save dir.
    :param global_step: 当前的迭代步 Current global training step #. Used in ckpt filename.
    """
    # Save model checkpoint
    # 设置保存路径
    output_dir = os.path.join(output_dir, "checkpoint-{}".format(global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    # 调用模型自身的 save_pretained 方法保存 checkpoint
    model_to_save.save_pretrained(output_dir)


def _save_model(model, output_dir, weights_name, config_name):
    """Save model to disk.
    保存模型到本地
    :param model: 需要保存的模型 Model to save (pytorch)
    :param output_dir: 保存路径 Path to model save dir.
    :param weights_name: 权重文件的名字 filename for model parameters.
    :param config_name: 设置文件的名字 filename for config.
    """
    model_to_save = model.module if hasattr(model, "module") else model

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(output_dir, weights_name)
    output_config_file = os.path.join(output_dir, config_name)

    # 保存模型的 state_dict()
    torch.save(model_to_save.state_dict(), output_model_file)
    # 保存模型的设置信息
    try:
        model_to_save.config.to_json_file(output_config_file)
    except AttributeError:
        # no config
        pass


def _get_eval_score(model, eval_dataloader):
    """Measure performance of a model on the evaluation set.
    对模型在测试集上的效果进行评价
    :param model: Model to test.
    :param eval_dataloader: a torch DataLoader that iterates through the eval set.

    :return: pearson correlation, if do_regression==True, else classification accuracy [0., 1.]
    """
    # 模型切换到测试模式
    model.eval()
    correct = 0 # 这变量好像没人用
    logits = []
    labels = []
    for input_ids, batch_labels in eval_dataloader:
        # 对每个 batch 进行测试
        batch_labels = batch_labels.to(device)
        if isinstance(input_ids, dict):
            ## dataloader collates dict backwards. This is a workaround to get
            # ids in the right shape for HuggingFace models
            input_ids = {k: torch.stack(v).T.to(device) for k, v in input_ids.items()}
            # import pdb
            # pdb.set_trace()
            with torch.no_grad():
                batch_logits = model(**input_ids)['logits']
        else:
            input_ids = input_ids.to(device)
            with torch.no_grad():
                batch_logits = model(input_ids)

        logits.extend(batch_logits.cpu().squeeze().tolist())
        labels.extend(batch_labels)

    # 切回训练模式
    model.train()
    # 预测结果与真实标签
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)

    # 如果是分类
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum()
    return float(correct) / len(labels) # 计算分类准确率

def _get_vibert_eval_score(model, eval_dataloader):
    """Measure performance of a model on the evaluation set.
    对模型在测试集上的效果进行评价
    :param model: Model to test.
    :param eval_dataloader: a torch DataLoader that iterates through the eval set.

    :return: pearson correlation, if do_regression==True, else classification accuracy [0., 1.]
    """
    # 模型切换到测试模式
    model.eval()
    correct = 0 # 这变量好像没人用
    logits = []
    labels = []
    for input_ids, batch_labels in eval_dataloader:
        # 对每个 batch 进行测试
        batch_labels = batch_labels.to(device)
        if isinstance(input_ids, dict):
            ## dataloader collates dict backwards. This is a workaround to get
            # ids in the right shape for HuggingFace models
            input_ids = {k: torch.stack(v).T.to(device) for k, v in input_ids.items()}
            # import pdb
            # pdb.set_trace()
            with torch.no_grad():
                batch_logits = model(input_ids)['logits']
        else:
            input_ids = input_ids.to(device)
            with torch.no_grad():
                batch_logits = model(input_ids)

        logits.extend(batch_logits.cpu().squeeze().tolist())
        labels.extend(batch_labels)

    # 切回训练模式
    model.train()
    # 预测结果与真实标签
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)

    # 如果是分类
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum()
    return float(correct) / len(labels) # 计算分类准确率

def _make_directories(output_dir):
    # 生成路径的帮助函数，若不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def _is_writable_type(obj):
    # tf board 记录 arg 的帮助函数，若 arg 的类型允许保存，返回真，否则返回假
    for ok_type in [bool, int, str, float]:
        if isinstance(obj, ok_type):
            return True
    return False


def batch_encode(tokenizer, text_list):
    # 对输入的文本进行分词
    if hasattr(tokenizer, "batch_encode"):
        # 如果有 batch_encode 方法
        return tokenizer.batch_encode(text_list) # 推测是返回一个 iterator
    else:
        # 如果分词器没用 batch encode 的方法
        return [tokenizer.encode(text_input) for text_input in text_list] # 对输入的 text list 全部进行分词并转换为字典下标
    # encode 方法应该还包含了添加特殊 token 的步骤

def _mixup_criterion(criterion, pred, y_a, y_b, lam):
    '''
    Mixup 模型的输出概率
    :param criterion:
    :param pred:
    :param y_a:
    :param y_b:
    :param lam:
    :return:
    '''
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def _batch_encoder(tokenizer, text):
    '''
    Large text list cause process killed. Orderly process
    :param tokenizer:
    :param text:
    :return:
    '''
    text_ids = []
    batch_number = len(text)//10000
    start, end = 0, 0
    for i in range(batch_number):
        start = i * 10000
        end = (i+1) * 10000
        text_ids.extend(batch_encode(tokenizer, text[start:end]))
    text_ids.extend(batch_encode(tokenizer, text[end:]))
    return text_ids

def _make_dataloader(tokenizer, text, labels, batch_size):
    """Create torch DataLoader from list of input text and labels.
    创建 Dataloader 类
    :param tokenizer: Tokenizer to use for this text.
    :param text: list of input text.
    :param labels: list of corresponding labels.
    :param batch_size: batch size (int).
    :return: torch DataLoader for this training set.
    """

    text_ids = _batch_encoder(tokenizer, text)  # 分词后转换为字典下标的文本
    input_ids = np.array(text_ids)  # 输入的字典 id 转化为 np 数组
    labels = np.array(labels)  # 对于标签也转换为 np 数组
    data = list((ids, label) for ids, label in zip(input_ids, labels)) # 转为一个 tuple 列表
    sampler = RandomSampler(data) # 设置随机采样器
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)  # 生成 Dataloader
    return dataloader



def train_model(args):
    train.shared.utils.set_seed(args.random_seed)  # 设置全局随机种子

    _make_directories(args.output_dir)  # 创建保存路径

    num_gpus = torch.cuda.device_count()  # 获取 GPU 数量
    num_gpus = 1

    # Save logger writes to file
    # 设置 logging 将日志保存至文件
    log_txt_path = os.path.join(args.output_dir, "log.txt") # 日志文件地址
    fh = logging.FileHandler(log_txt_path) # 创建 logging，具体用法不了解
    fh.setLevel(logging.DEBUG) # 设置 log 级别，看样子是
    logger.addHandler(fh)  # 这也不太懂
    logger.info(f"Writing logs to {log_txt_path}.") # 输出 log 句

    # Get list of text and list of label (integers) from disk.
    # 获取文本与标签
    train_text, train_labels, eval_text, eval_labels = dataset_from_local(args) #读取本地数据  # dataset_from_args(args) # 根据传入的参数读取对应的数据集，返回的训练集、测试集的文本和标签 list

    # 如果设置了使用数据集的百分比，获取对应百分比的训练数据  TODO 把这个改到 adversarial example 上
    if args.pct_dataset < 1.0:
        logger.info(f"Using {args.pct_dataset*100}% of the training set")
        (train_text, train_labels), _ = _train_val_split(
            train_text, train_labels, split_val=1.0 - args.pct_dataset
        )
    train_examples_len = len(train_text)

    # 打印输出日志
    label_set = set(train_labels) # 获取标签集合
    args.num_labels = len(label_set) # 获取标签数量
    logger.info(f"Loaded dataset. Found: {args.num_labels} labels: {sorted(label_set)}") # 记录到日志

    # 对数据集的长度进行检验，查看文本数量与标签数量是否一致
    if len(train_labels) != len(train_text):
        raise ValueError(
            f"Number of train examples ({len(train_text)}) does not match number of labels ({len(train_labels)})"
        )
    if len(eval_labels) != len(eval_text):
        raise ValueError(
            f"Number of teste xamples ({len(eval_text)}) does not match number of labels ({len(eval_labels)})"
        )

    # 根据参数载入对应的模型和分词器
    model_wrapper = model_from_args(args, args.num_labels) # 一个神奇的函数
    model = model_wrapper.model # 获取模型
    tokenizer = model_wrapper.tokenizer # 获取分词器

    # multi-gpu training
    # 多 GPU 训练
    if num_gpus > 1:  # 若 GPU 数量大于 1
        model = torch.nn.DataParallel(model)  #使用 torch 内置的多 GPU 方式
        logger.info("Using torch.nn.DataParallel.")
    logger.info(f"Training model across {num_gpus} GPUs")

    num_train_optimization_steps = (
        int(train_examples_len / args.batch_size / args.grad_accum_steps)
        * args.num_train_epochs
    )  # 计算需要更新的步数

    # 根据不同的模型，设置需要更新的参数与对应的优化器
    if args.model == "lstm" or args.model == "cnn":
        # 如果是 lstm 模型或者 cnn 模型
        def need_grad(x):
            return x.requires_grad
        # 定义判断是否需要梯度更新的函数，使用 filter 对模型参数进行过滤， filter 函数返回判断函数结果为 true 的 iterator
        optimizer = torch.optim.Adam( # 设置 Adam 优化器
            filter(need_grad, model.parameters()), lr=args.learning_rate
        )
        scheduler = None  # 没有学习率的更新
    else:
        # 如果是 bert-based 模型
        param_optimizer = list(model.named_parameters()) # 获取所有参数
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"] # 对于 bias, Normalization 层的参数不设置正则约束
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay) # 任何参数都不在无需正则的参数中
                ],
                "weight_decay": 0.01, # 梯度衰减设置为 0.01
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay) #对于任何在不需要正则的参数中
                ],
                "weight_decay": 0.0, # 梯度衰减设置为 0.0
            },
        ]

        optimizer = transformers.optimization.AdamW( # AdamW 优化器，传入需要优化的参数，设置学习率
            optimizer_grouped_parameters, lr=args.learning_rate
        )

        scheduler = transformers.optimization.get_linear_schedule_with_warmup( # 优化器的学习率使用 warmup 策略进行更新
            optimizer,   # 优化器
            num_warmup_steps=args.warmup_proportion, # warmup 开始的比例
            num_training_steps=num_train_optimization_steps,  # 总的优化步数
        )

    # Start Tensorboard and log hyperparams.
    # 引入 tensorboard 记录模型学习的日志
    from torch.utils.tensorboard import SummaryWriter

    tb_writer = SummaryWriter(args.output_dir) # 创建 tneosrboard 记录的 writer，对 wirter 传入保存路径

    # Use Weights & Biases, if enabled.
    # 若要记录权重和偏置项的更新
    if args.enable_wandb:
        global wandb
        wandb = train.shared.utils.LazyLoader("wandb", globals(), "wandb")

        wandb.init(sync_tensorboard=True)

    # Save original args to file
    # 将训练参数保存到 json 文件
    args_save_path = os.path.join(args.output_dir, "train_args.json")
    _save_args(args, args_save_path)
    logger.info(f"Wrote original training args to {args_save_path}.")

    tb_writer.add_hparams(
        {k: v for k, v in vars(args).items() if _is_writable_type(v)}, {}
    ) # 将所有可以保存的参数添加到 writer

    # 评价用的 dataloader
    eval_dataloader = _make_dataloader(
        tokenizer, eval_text, eval_labels, args.batch_size # 返回 torch 的标准 Dataloader：输入为 text id，输出 label
    )
    # 训练用的 dataloader
    train_dataloader = _make_dataloader(
        tokenizer, train_text, train_labels, args.batch_size
    )
    # 初始化一些参数
    global_step = 0 # 全局优化步从 0 开始
    tr_loss = 0 # tr 是啥，不清楚啥 loss

    # 设置模型为训练模式
    model.train()
    args.best_eval_score = 0 # 当前最好的评价分数
    args.best_eval_score_epoch = 0 # 最好评价分数对应的训练轮数
    args.epochs_since_best_eval_score = 0 # 达到当前最好分数后又训练了的轮数，或许是用于 early stopping 的

    # 设置 loss 的计算
    def loss_backward(loss):
        # 如果是多 GPU 训练
        if num_gpus > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training， 获取多个 gpu 上训练的平均损失
        if args.grad_accum_steps > 1: # 如果设置了梯度累加步
            loss = loss / args.grad_accum_steps # 对累加后的梯度还原
        loss.backward() # 反传 loss
        return loss

    # 损失
    loss_fct = torch.nn.CrossEntropyLoss() # 交叉熵损失


    if args.adversarial_training:
        # 仅在设置的对抗训练周期中使用对抗训练样本
        logger.info(f"Read perturbed dataset from file {args.file_paths}")
        adv_dataset = PerturbedDataset(args.file_paths, tokenizer)
        perturbed_text, perturbed_label = adv_dataset.perturbed_string()
        train_dataloader = _make_dataloader(
            tokenizer, train_text+perturbed_text, train_labels+perturbed_label, args.batch_size
        )
        train_examples_len = len(train_text+adv_dataset.perturbed_list)
    if args.mixup_training:
        logger.info(f"Read perturbed dataset from file {args.file_paths}")
        adv_dataset = PerturbedDataset(args.file_paths, tokenizer)
        adv_dataloader = DataLoader(adv_dataset, shuffle=True, batch_size=args.batch_size)
        train_dataloader_b = _make_dataloader(
            tokenizer, train_text, train_labels, args.batch_size
        )
        train_examples_len = len(adv_dataset.perturbed_list)

    # import pdb
    # pdb.set_trace()
    # Start training
    # 开始训练
    print(model)
    logger.info("***** Running training *****")
    logger.info(f"\tNum examples = {train_examples_len}")
    logger.info(f"\tBatch size = {args.batch_size}")
    logger.info(f"\tMax sequence length = {args.max_length}")
    logger.info(f"\tNum steps = {num_train_optimization_steps}")
    logger.info(f"\tNum epochs = {args.num_train_epochs}")
    logger.info(f"\tLearning rate = {args.learning_rate}")

    mix_weight = args.mix_weight if args.mix_weight else 6
    # 进行每轮迭代
    for epoch in tqdm.trange(
        int(args.num_train_epochs), desc="Epoch", position=0, leave=True
    ):
        if args.mixup_training:
            # 对抗训练
            prog_bar = tqdm.tqdm(adv_dataloader, desc="Iteration", position=0, leave=True)  # 进度条

            # Use these variables to track training accuracy during classification.
            correct_predictions = 0  # 正确分类的对抗样本
            total_predictions = 0  # 总共正确分类的对抗样本
            # run step
            for step, batch in enumerate(zip(train_dataloader, prog_bar, train_dataloader_b)):
                normal_batch, adv_batch, normal_batch_b = batch

                origin_ids_a, perturb_ids, labels_a = adv_batch  # 对抗样本
                origin_ids_a = {k: torch.stack(v).T.to(device) for k, v in origin_ids_a.items()} if isinstance(origin_ids_a, dict) else origin_ids_a.to(device)# 对 tokenizer 的 dict 都转到 device 上
                labels_a = labels_a.to(device)
                perturb_ids = {k: torch.stack(v).T.to(device) for k, v in perturb_ids.items()} if isinstance(perturb_ids, dict) else perturb_ids.to(device)

                # 正常输入
                logits = model(origin_ids_a)  # 获取普通样本的 logits
                loss = loss_fct(logits, labels_a)  # 普通样本的 loss
                m = np.float32(np.random.beta(1, 1))   # mixup ratio
                # 在成对的普通样本与对抗样本进行 mixup，使用 JS 损失

                if args.regularized_adv_example:
                    if args.adv_mixup:
                        logits_mix = model(_input=origin_ids_a, perturbed_input=perturb_ids, mix_ratio=m)  # 获取 mixup 样本的 logits，多输入不传 Position batch_size=1 时报输入缺失错误
                        prob_logits, prob_mix = F.softmax(logits, dim=1), F.softmax(logits_mix, dim=1)
                        p_mixture = torch.clamp((prob_logits + prob_mix) / 2., 1e-7, 1).log()  # 普通样本与对抗样本的平均 prob
                        loss +=  mix_weight * (F.kl_div(p_mixture, prob_logits, reduction='batchmean') +   # 默认为 6，手工调整零时调整测试，没有用参数控制
                                     F.kl_div(p_mixture, prob_mix, reduction='batchmean')) / 2.  # JS 散度
                    else:
                        # loss count on adv example
                        logits_mix = model(perturb_ids)  # 获取 mixup 样本的 logits，多输入不传 Position batch_size=1 时报输入缺失错误
                        prob_logits, prob_mix = F.softmax(logits, dim=1), F.softmax(logits_mix, dim=1)
                        p_mixture = torch.clamp((prob_logits + prob_mix) / 2., 1e-7, 1).log()  # 普通样本与对抗样本的平均 prob
                        loss +=  mix_weight * (F.kl_div(p_mixture, prob_logits, reduction='batchmean') +   # 默认为 6，手工调整零时调整测试，没有用参数控制
                                     F.kl_div(p_mixture, prob_mix, reduction='batchmean')) / 2.  # JS 散度


                if args.mix_normal_example:
                    # 在普通样本间也进行 mixup，使用普通 mixup 的计算方法
                    origin_ids_n, labels_n = normal_batch
                    origin_ids_b, labels_b = normal_batch_b  # 普通样本
                    # if labels_a.shape != labels_b.shape:
                    #     continue
                    labels_n = labels_n.to(device)
                    origin_ids_n = {k: torch.stack(v).T.to(device) for k, v in origin_ids_n.items()} if isinstance(origin_ids_n, dict) else origin_ids_n.to(device)
                    labels_b = labels_b.to(device)
                    origin_ids_b = {k: torch.stack(v).T.to(device) for k, v in origin_ids_b.items()}if isinstance(origin_ids_b, dict) else origin_ids_b.to(device)
                    logits_normal_mix = model(_input=origin_ids_n, perturbed_input=origin_ids_b, mix_ratio=m)  # 普通样本之间进行 mixup
                    loss += _mixup_criterion(loss_fct, logits_normal_mix, labels_n, labels_b, m)

                # 计算正确分类的样本数量
                pred_labels = logits.argmax(dim=-1)
                correct_predictions += (pred_labels == labels_a).sum().item()
                total_predictions += len(pred_labels)

                # 损失反传
                loss = loss_backward(loss)
                tr_loss += loss.item()  # 损失累加

                # 如果为记录日志的训练步
                if global_step % args.tb_writer_step == 0:
                    tb_writer.add_scalar("loss", loss.item(), global_step)  # 记录损失
                    # 记录学习率
                    if scheduler is not None:
                        tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                    else:
                        tb_writer.add_scalar("lr", args.learning_rate, global_step)

                if global_step > 0:
                    prog_bar.set_description(f"Epoch {epoch} Loss {tr_loss / global_step}")  # 更新进度条的展示
                if (step + 1) % args.grad_accum_steps == 0:
                    optimizer.step()  # 进行优化
                    # 若使用了学习率的控制器
                    if scheduler is not None:
                        scheduler.step()  # 更新学习率
                    optimizer.zero_grad()  # 梯度重置为 0
                # Save model checkpoint to file.
                # 保存模型
                if (
                        global_step > 0
                        and (args.checkpoint_steps > 0)
                        and (global_step % args.checkpoint_steps) == 0
                ):
                    _save_model_checkpoint(model, args.output_dir, global_step)

                # Inc step counter.
                # 全局训练步数加一
                global_step += 1
        else:
            # 普通的 training 或 adversarial training
            prog_bar = tqdm.tqdm(train_dataloader, desc="Iteration", position=0, leave=True)  # 进度条

            # Use these variables to track training accuracy during classification.
            correct_predictions = 0 # 正确分类的对抗样本
            total_predictions = 0 # 总共正确分类的对抗样本
            # 训练
            for step, batch in enumerate(prog_bar):
                input_ids, labels = batch

                labels = labels.to(device)
                # import pdb
                # pdb.set_trace()
                if isinstance(input_ids, dict):
                    # 如果传入的 input_ids 是个字典，将字典转为 huggingface 模型需要的输入形状
                    ## dataloader collates dict backwards. This is a workaround to get
                    # ids in the right shape for HuggingFace models
                    input_ids = {
                        k: torch.stack(v).T.to(device) for k, v in input_ids.items()
                    }
                    logits = model(**input_ids)['logits']
                else:
                    # 正常输入
                    input_ids = input_ids.to(device)
                    logits = model(input_ids) # 获取模型输出

                # 使用交叉熵损失

                loss = loss_fct(logits, labels)
                # 计算正确分类的样本数量
                pred_labels = logits.argmax(dim=-1)
                correct_predictions += (pred_labels == labels).sum().item()
                total_predictions += len(pred_labels)

                # 损失反传
                loss = loss_backward(loss)
                tr_loss += loss.item()  # 损失累加

                # 如果为记录日志的训练步
                if global_step % args.tb_writer_step == 0:
                    tb_writer.add_scalar("loss", loss.item(), global_step)  # 记录损失
                    # 记录学习率
                    if scheduler is not None:
                        tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                    else:
                        tb_writer.add_scalar("lr", args.learning_rate, global_step)

                if global_step > 0:
                    prog_bar.set_description(f"Epoch {epoch} Loss {tr_loss/global_step}")  # 更新进度条的展示
                if (step + 1) % args.grad_accum_steps == 0:
                    optimizer.step() # 进行优化
                    # 若使用了学习率的控制器
                    if scheduler is not None:
                        scheduler.step()  # 更新学习率
                    optimizer.zero_grad()  # 梯度重置为 0
                # Save model checkpoint to file.
                # 保存模型
                if (
                    global_step > 0
                    and (args.checkpoint_steps > 0)
                    and (global_step % args.checkpoint_steps) == 0
                ):
                    _save_model_checkpoint(model, args.output_dir, global_step)

                # Inc step counter.
                # 全局训练步数加一
                global_step += 1

        _save_model(model, args.output_dir, args.weights_name, args.config_name)
        # Print training accuracy, if we're tracking it.
        # 记录训练的准确率
        if total_predictions > 0:
            train_acc = correct_predictions / total_predictions
            logger.info(f"Train accuracy: {train_acc*100}%")
            tb_writer.add_scalar("epoch_train_score", train_acc, epoch)

        # Check accuracy after each epoch.
        # skip args.num_clean_epochs during adversarial training
        # 如果为正常训练，或对抗训练中使用了对抗样本的轮次，则进行准确性验证
        if (epoch >= args.num_clean_epochs):
            eval_score = _get_eval_score(model, eval_dataloader)
            tb_writer.add_scalar("epoch_eval_score", eval_score, epoch)

            # 如果每轮都要保存模型
            if args.checkpoint_every_epoch:
                _save_model_checkpoint(model, args.output_dir, args.global_step)

            # 输出日志
            logger.info(
                f"Eval accuracy: {eval_score*100}%"
            )
            # 如果训练后的分数更好了，更新最好结果；否则检查是否达到了 early stopping 的轮数，提前终止训练
            if eval_score > args.best_eval_score:
                args.best_eval_score = eval_score
                args.best_eval_score_epoch = epoch
                args.epochs_since_best_eval_score = 0
                _save_model(model, args.output_dir, args.weights_name, args.config_name)  # 每次保存最优模型
                logger.info(f"Best acc found. Saved model to {args.output_dir}.")
                _save_args(args, args_save_path)
                logger.info(f"Saved updated args to {args_save_path}")
            else:
                args.epochs_since_best_eval_score += 1
                if (args.early_stopping_epochs > 0) and (
                    args.epochs_since_best_eval_score > args.early_stopping_epochs
                ):
                    logger.info(
                        f"Stopping early since it's been {args.early_stopping_epochs} epochs since validation acc increased"
                    )
                    break

    # 如果保存最后一次训练的模型
    if args.save_last:
        _save_model(model, args.output_dir, args.weights_name, args.config_name)

    # read the saved model and report its eval performance
    # 结束所有训练后，载入保存的模型，进行测试
    logger.info("Finished training. Re-loading and evaluating model from disk.")
    model_wrapper = model_from_args(args, args.num_labels)  # model_wrappper 不是从保存的 json 读出来的，而是直接读传入的 args
    model = model_wrapper.model
    model.load_state_dict(torch.load(os.path.join(args.output_dir, args.weights_name)))  # 载入模型
    eval_score = _get_eval_score(model, eval_dataloader)  # 在正常测试数据上进行评价

    # 如果保存最后一次训练的模型
    if args.save_last:
        args.best_eval_score = eval_score
        args.best_eval_score_epoch = epoch
    # 记录测试样本上的准确率
    logger.info(
        f"Saved model accuracy: {eval_score*100}%"
    )

    # end of training, save tokenizer
    # 训练结束，保存分词器
    try:
        tokenizer.save_pretrained(args.output_dir)
        logger.info(f"Saved tokenizer {tokenizer} to {args.output_dir}.")
    except AttributeError:
        logger.warn(
            f"Error: could not save tokenizer {tokenizer} to {args.output_dir}."
        )

    # Save a little readme with model info
    # 生成并保存简短的模型说明文件
    write_readme(args, args.best_eval_score, args.best_eval_score_epoch)

    _save_args(args, args_save_path)
    tb_writer.close()
    logger.info(f"Wrote final training args to {args_save_path}.")


def train_vib(args):
    train.shared.utils.set_seed(args.random_seed)  # 设置全局随机种子

    _make_directories(args.output_dir)  # 创建保存路径

    num_gpus = torch.cuda.device_count()  # 获取 GPU 数量
    num_gpus = 1

    # Save logger writes to file
    # 设置 logging 将日志保存至文件
    log_txt_path = os.path.join(args.output_dir, "log.txt") # 日志文件地址
    fh = logging.FileHandler(log_txt_path) # 创建 logging，具体用法不了解
    fh.setLevel(logging.DEBUG) # 设置 log 级别，看样子是
    logger.addHandler(fh)  # 这也不太懂
    logger.info(f"Writing logs to {log_txt_path}.") # 输出 log 句

    # Get list of text and list of label (integers) from disk.
    # 获取文本与标签
    train_text, train_labels, eval_text, eval_labels = dataset_from_local(args) #读取本地数据  # dataset_from_args(args) # 根据传入的参数读取对应的数据集，返回的训练集、测试集的文本和标签 list

    # 如果设置了使用数据集的百分比，获取对应百分比的训练数据  TODO 把这个改到 adversarial example 上
    if args.pct_dataset < 1.0:
        logger.info(f"Using {args.pct_dataset*100}% of the training set")
        (train_text, train_labels), _ = _train_val_split(
            train_text, train_labels, split_val=1.0 - args.pct_dataset
        )
    train_examples_len = len(train_text)

    # 打印输出日志
    label_set = set(train_labels) # 获取标签集合
    args.num_labels = len(label_set) # 获取标签数量
    logger.info(f"Loaded dataset. Found: {args.num_labels} labels: {sorted(label_set)}") # 记录到日志

    # 对数据集的长度进行检验，查看文本数量与标签数量是否一致
    if len(train_labels) != len(train_text):
        raise ValueError(
            f"Number of train examples ({len(train_text)}) does not match number of labels ({len(train_labels)})"
        )
    if len(eval_labels) != len(eval_text):
        raise ValueError(
            f"Number of teste xamples ({len(eval_text)}) does not match number of labels ({len(eval_labels)})"
        )

    # 根据参数载入对应的模型和分词器
    model_wrapper = model_from_args(args, args.num_labels) # 一个神奇的函数
    model = model_wrapper.model # 获取模型
    tokenizer = model_wrapper.tokenizer # 获取分词器

    # multi-gpu training
    # 多 GPU 训练
    if num_gpus > 1:  # 若 GPU 数量大于 1
        model = torch.nn.DataParallel(model)  #使用 torch 内置的多 GPU 方式
        logger.info("Using torch.nn.DataParallel.")
    logger.info(f"Training model across {num_gpus} GPUs")

    num_train_optimization_steps = (
        int(train_examples_len / args.batch_size / args.grad_accum_steps)
        * args.num_train_epochs
    )  # 计算需要更新的步数

    # 如果是 bert-based 模型
    param_optimizer = list(model.named_parameters()) # 获取所有参数
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"] # 对于 bias, Normalization 层的参数不设置正则约束
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay) # 任何参数都不在无需正则的参数中
            ],
            "weight_decay": 0.01, # 梯度衰减设置为 0.01
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay) #对于任何在不需要正则的参数中
            ],
            "weight_decay": 0.0, # 梯度衰减设置为 0.0
        },
    ]

    optimizer = transformers.optimization.AdamW( # AdamW 优化器，传入需要优化的参数，设置学习率
        optimizer_grouped_parameters, lr=args.learning_rate
    )

    scheduler = transformers.optimization.get_linear_schedule_with_warmup( # 优化器的学习率使用 warmup 策略进行更新
        optimizer,   # 优化器
        num_warmup_steps=args.warmup_proportion, # warmup 开始的比例
        num_training_steps=num_train_optimization_steps,  # 总的优化步数
    )

    # Start Tensorboard and log hyperparams.
    # 引入 tensorboard 记录模型学习的日志
    from torch.utils.tensorboard import SummaryWriter

    tb_writer = SummaryWriter(args.output_dir) # 创建 tneosrboard 记录的 writer，对 wirter 传入保存路径

    # Use Weights & Biases, if enabled.
    # 若要记录权重和偏置项的更新
    if args.enable_wandb:
        global wandb
        wandb = train.shared.utils.LazyLoader("wandb", globals(), "wandb")

        wandb.init(sync_tensorboard=True)

    # Save original args to file
    # 将训练参数保存到 json 文件
    args_save_path = os.path.join(args.output_dir, "train_args.json")
    _save_args(args, args_save_path)
    logger.info(f"Wrote original training args to {args_save_path}.")

    tb_writer.add_hparams(
        {k: v for k, v in vars(args).items() if _is_writable_type(v)}, {}
    ) # 将所有可以保存的参数添加到 writer

    # 评价用的 dataloader
    eval_dataloader = _make_dataloader(
        tokenizer, eval_text, eval_labels, args.batch_size # 返回 torch 的标准 Dataloader：输入为 text id，输出 label
    )
    # 训练用的 dataloader
    train_dataloader = _make_dataloader(
        tokenizer, train_text, train_labels, args.batch_size
    )
    # 初始化一些参数
    global_step = 0 # 全局优化步从 0 开始
    tr_loss = 0 # tr 是啥，不清楚啥 loss

    # 设置模型为训练模式
    model.train()
    args.best_eval_score = 0 # 当前最好的评价分数
    args.best_eval_score_epoch = 0 # 最好评价分数对应的训练轮数
    args.epochs_since_best_eval_score = 0 # 达到当前最好分数后又训练了的轮数，或许是用于 early stopping 的

    # 设置 loss 的计算
    def loss_backward(loss):
        # 如果是多 GPU 训练
        if num_gpus > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training， 获取多个 gpu 上训练的平均损失
        if args.grad_accum_steps > 1: # 如果设置了梯度累加步
            loss = loss / args.grad_accum_steps # 对累加后的梯度还原
        loss.backward() # 反传 loss
        return loss

    # 损失
    loss_fct = torch.nn.CrossEntropyLoss() # 交叉熵损失

    # import pdb
    # pdb.set_trace()
    # Start training
    # 开始训练
    # print(model)
    logger.info("***** Running training *****")
    logger.info(f"\tNum examples = {train_examples_len}")
    logger.info(f"\tBatch size = {args.batch_size}")
    logger.info(f"\tMax sequence length = {args.max_length}")
    logger.info(f"\tNum steps = {num_train_optimization_steps}")
    logger.info(f"\tNum epochs = {args.num_train_epochs}")
    logger.info(f"\tLearning rate = {args.learning_rate}")
    logger.info(f"\tBeta = {args.beta}")

    # 进行每轮迭代
    for epoch in tqdm.trange(
        1, int(args.num_train_epochs), desc="Epoch", position=0, leave=True
    ):
        # 普通的 training 或 adversarial training
        prog_bar = tqdm.tqdm(train_dataloader, desc="Iteration", position=0, leave=True)  # 进度条

        # Use these variables to track training accuracy during classification.
        correct_predictions = 0 # 正确分类的对抗样本
        total_predictions = 0 # 总共正确分类的对抗样本
        # 训练
        for step, batch in enumerate(prog_bar):
            input_ids, labels = batch

            labels = labels.to(device)
            # import pdb
            # pdb.set_trace()
            inputs = {
                k: torch.stack(v).T.to(device) for k, v in input_ids.items()
            }
            # outputs = model(**inputs)
            outputs = model(inputs=inputs, labels=labels, epoch=epoch, sampling_type="idd")
            loss = outputs["loss"]  # model outputs are always tuple in transformers (see doc)
            logits = outputs["logits"]

            # 使用交叉生损失

            # loss = loss_fct(logits, labels)
            # 计算正确分类的样本数量
            pred_labels = logits.argmax(dim=-1)
            correct_predictions += (pred_labels == labels).sum().item()
            total_predictions += len(pred_labels)

            # 损失反传
            loss = loss_backward(loss)
            tr_loss += loss.item()  # 损失累加

            # 如果为记录日志的训练步
            if global_step % args.tb_writer_step == 0:
                tb_writer.add_scalar("loss", loss.item(), global_step)  # 记录损失
                # 记录学习率
                if scheduler is not None:
                    tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                else:
                    tb_writer.add_scalar("lr", args.learning_rate, global_step)

            if global_step > 0:
                prog_bar.set_description(f"Epoch {epoch} Loss {tr_loss/global_step}")  # 更新进度条的展示
            if (step + 1) % args.grad_accum_steps == 0:
                optimizer.step() # 进行优化
                # 若使用了学习率的控制器
                if scheduler is not None:
                    scheduler.step()  # 更新学习率
                optimizer.zero_grad()  # 梯度重置为 0
            # Save model checkpoint to file.
            # 保存模型
            if (
                global_step > 0
                and (args.checkpoint_steps > 0)
                and (global_step % args.checkpoint_steps) == 0
            ):
                _save_model_checkpoint(model, args.output_dir, global_step)

            # Inc step counter.
            # 全局训练步数加一
            global_step += 1

        # Print training accuracy, if we're tracking it.
        # 记录训练的准确率
        if total_predictions > 0:
            train_acc = correct_predictions / total_predictions
            logger.info(f"Train accuracy: {train_acc*100}%")
            tb_writer.add_scalar("epoch_train_score", train_acc, epoch)

        # Check accuracy after each epoch.
        # skip args.num_clean_epochs during adversarial training
        # 如果为正常训练，或对抗训练中使用了对抗样本的轮次，则进行准确性验证
        if (epoch >= args.num_clean_epochs):
            eval_score = _get_vibert_eval_score(model, eval_dataloader)
            tb_writer.add_scalar("epoch_eval_score", eval_score, epoch)

            # 如果每轮都要保存模型
            if args.checkpoint_every_epoch:
                _save_model_checkpoint(model, args.output_dir, args.global_step)

            # 输出日志
            logger.info(
                f"Eval accuracy: {eval_score*100}%"
            )
            # 如果训练后的分数更好了，更新最好结果；否则检查是否达到了 early stopping 的轮数，提前终止训练
            if eval_score > args.best_eval_score:
                args.best_eval_score = eval_score
                args.best_eval_score_epoch = epoch
                args.epochs_since_best_eval_score = 0
                _save_model(model, args.output_dir, args.weights_name, args.config_name)  # 每次保存最优模型
                logger.info(f"Best acc found. Saved model to {args.output_dir}.")
                _save_args(args, args_save_path)
                logger.info(f"Saved updated args to {args_save_path}")
            else:
                args.epochs_since_best_eval_score += 1
                if (args.early_stopping_epochs > 0) and (
                    args.epochs_since_best_eval_score > args.early_stopping_epochs
                ):
                    logger.info(
                        f"Stopping early since it's been {args.early_stopping_epochs} epochs since validation acc increased"
                    )
                    break

    # 如果保存最后一次训练的模型
    if args.save_last:
        _save_model(model, args.output_dir, args.weights_name, args.config_name)

    # read the saved model and report its eval performance
    # 结束所有训练后，载入保存的模型，进行测试
    logger.info("Finished training. Re-loading and evaluating model from disk.")
    model_wrapper = model_from_args(args, args.num_labels)  # model_wrappper 不是从保存的 json 读出来的，而是直接读传入的 args
    model = model_wrapper.model
    model.load_state_dict(torch.load(os.path.join(args.output_dir, args.weights_name)))  # 载入模型
    eval_score = _get_eval_score(model, eval_dataloader)  # 在正常测试数据上进行评价

    # 如果保存最后一次训练的模型
    if args.save_last:
        args.best_eval_score = eval_score
        args.best_eval_score_epoch = epoch
    # 记录测试样本上的准确率
    logger.info(
        f"Saved model accuracy: {eval_score*100}%"
    )

    # end of training, save tokenizer
    # 训练结束，保存分词器
    try:
        tokenizer.save_pretrained(args.output_dir)
        logger.info(f"Saved tokenizer {tokenizer} to {args.output_dir}.")
    except AttributeError:
        logger.warn(
            f"Error: could not save tokenizer {tokenizer} to {args.output_dir}."
        )

    # Save a little readme with model info
    # 生成并保存简短的模型说明文件
    write_readme(args, args.best_eval_score, args.best_eval_score_epoch)

    _save_args(args, args_save_path)
    tb_writer.close()
    logger.info(f"Wrote final training args to {args_save_path}.")