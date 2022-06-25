import collections
import json
import logging
import math
import os
import random

import numpy as np
import scipy
import torch
from torch.utils.data import DataLoader, RandomSampler
import tqdm
import transformers

import textattack

from .train_args_helpers import (
    attack_from_args,
    augmenter_from_args,
    dataset_from_args,
    model_from_args,
    write_readme,
)

device = textattack.shared.utils.device
logger = textattack.shared.logger


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


def _filter_labels(text, labels, allowed_labels):
    """Keep examples with approved labels.
    根据 allowed_labels 标签筛选样本
    :param text: 样本文本列表list of text inputs.
    :param labels: 对应的标签 list of corresponding labels.
    :param allowed_labels: 要保留的标签 list of approved label values.

    :return: (final_text, final_labels). Filtered version of text and labels
    """
    final_text, final_labels = [], []
    for text, label in zip(text, labels):
        if label in allowed_labels:
            final_text.append(text)
            final_labels.append(label)
    return final_text, final_labels


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


def _get_eval_score(model, eval_dataloader, do_regression):
    """Measure performance of a model on the evaluation set.
    对模型在测试集上的效果进行评价
    :param model: Model to test.
    :param eval_dataloader: a torch DataLoader that iterates through the eval set.
    :param do_regression: bool. Whether we are doing regression (True) or classification (False)

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
            with torch.no_grad():
                batch_logits = model(**input_ids)[0]
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

    # 如果是回归
    if do_regression:
        # 计算关联系数
        pearson_correlation, pearson_p_value = scipy.stats.pearsonr(logits, labels)
        return pearson_correlation
    else:
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
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def _make_dataloader(tokenizer, text, labels, batch_size):
    """Create torch DataLoader from list of input text and labels.
    创建 Dataloader 类
    :param tokenizer: Tokenizer to use for this text.
    :param text: list of input text.
    :param labels: list of corresponding labels.
    :param batch_size: batch size (int).
    :return: torch DataLoader for this training set.
    """
    text_ids = batch_encode(tokenizer, text)  # 分词后转换为字典下标的文本
    input_ids = np.array(text_ids)  # 输入的字典 id 转化为 np 数组
    labels = np.array(labels)  # 对于标签也转换为 np 数组
    data = list((ids, label) for ids, label in zip(input_ids, labels)) # 转为一个 tuple 列表
    sampler = RandomSampler(data) # 设置随机采样器
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)  # 生成 Dataloader
    return dataloader


def _data_augmentation(text, labels, augmenter):
    """Use an augmentation method to expand a training set.
    使用数据增强方法对输入的文本进行增强
    :param text: list of input text.
    :param labels: list of corresponding labels.
    :param augmenter: textattack.augmentation.Augmenter, augmentation scheme.

    :return: augmented_text, augmented_labels. list of (augmented) input text and labels.
    """
    aug_text = augmenter.augment_many(text)
    # flatten augmented examples and duplicate labels
    flat_aug_text = []
    flat_aug_labels = []
    for i, examples in enumerate(aug_text):
        for aug_ver in examples:
            flat_aug_text.append(aug_ver)
            flat_aug_labels.append(labels[i])
    return flat_aug_text, flat_aug_labels


def _generate_adversarial_examples(model, attack_class, dataset):
    """Create a dataset of adversarial examples based on perturbations of the
    existing dataset.
    对于传入的模型与设置的攻击策略，对于传入的数据集生成对抗样本

    :param model: Model to attacks.
    :param attack_class: class name of attacks recipe to run.
    :param dataset: iterable of (text, label) pairs.

    :return: list(AttackResult) of adversarial examples.
    """
    # 创建对抗策略
    attack = attack_class.build(model)

    # 这段对 try tensorflow GPU 内存进行设置
    try:
        # Fix TensorFlow GPU memory growth
        import tensorflow as tf

        tf.get_logger().setLevel("WARNING")
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
    except ModuleNotFoundError:
        pass

    # 直接生成对抗样本
    adv_attack_results = []
    for adv_ex in tqdm.tqdm(
        attack.attack_dataset(dataset), desc="Attack", total=len(dataset)
    ):
        adv_attack_results.append(adv_ex)
    # 返回结果
    return adv_attack_results


def train_model(args):
    textattack.shared.utils.set_seed(args.random_seed) # 设置全局随机种子？

    logger.warn(
        "WARNING: TextAttack's model training feature is in beta. Please report any issues on our Github page, https://github.com/QData/TextAttack/issues."
    )
    _make_directories(args.output_dir) # 创建保存路径

    num_gpus = torch.cuda.device_count() # 获取 GPU 数量

    # Save logger writes to file
    # 设置 logging 将日志保存至文件
    log_txt_path = os.path.join(args.output_dir, "log.txt") # 日志文件地址
    fh = logging.FileHandler(log_txt_path) # 创建 logging，具体用法不了解
    fh.setLevel(logging.DEBUG) # 设置 log 级别，看样子是
    logger.addHandler(fh) # 这也不太懂
    logger.info(f"Writing logs to {log_txt_path}.") # 输出 log 句

    # Get list of text and list of label (integers) from disk.
    # 从本地获取文本与标签
    train_text, train_labels, eval_text, eval_labels = dataset_from_args(args) # 根据传入的参数读取对应的数据集，返回的训练集、测试集的文本和标签 list

    # Filter labels
    # 根据 allowed_label，过滤数据集，仅 allowed_label 对应的数据集会保留
    if args.allowed_labels:
        train_text, train_labels = _filter_labels(
            train_text, train_labels, args.allowed_labels
        )
        eval_text, eval_labels = _filter_labels(
            eval_text, eval_labels, args.allowed_labels
        )

    # 如果设置了使用数据集的百分比，获取对应百分比的训练数据
    if args.pct_dataset < 1.0:
        logger.info(f"Using {args.pct_dataset*100}% of the training set")
        (train_text, train_labels), _ = _train_val_split(
            train_text, train_labels, split_val=1.0 - args.pct_dataset
        )
    train_examples_len = len(train_text)

    # data augmentation
    # 获取数据增强器 augmenter
    augmenter = augmenter_from_args(args)
    if augmenter:
        logger.info(f"Augmenting {len(train_text)} samples with {augmenter}")
        # 对训练数据进行增强
        train_text, train_labels = _data_augmentation(
            train_text, train_labels, augmenter
        )

    # label_id_len = len(train_labels)
    # 打印输出日志
    label_set = set(train_labels) # 获取标签集合
    args.num_labels = len(label_set) # 获取标签数量
    logger.info(f"Loaded dataset. Found: {args.num_labels} labels: {sorted(label_set)}") # 记录到日志

    # 根据标签是否为浮点数，设置是否使用回归的 flag
    if isinstance(train_labels[0], float):
        # 若是浮点数，do_regression 设置为 True
        # TODO come up with a more sophisticated scheme for knowing when to do regression
        logger.warn("Detected float labels. Doing regression.")
        args.num_labels = 1
        args.do_regression = True
    else:
        # 若不是浮点数，do_regression 设置为 False
        args.do_regression = False

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

    attack_class = attack_from_args(args) # 获取 attacks
    # We are adversarial training if the user specified an attacks along with
    # the training args.
    # 如果设置了 adversarial training 在参数中，则进行进行 AT
    adversarial_training = (attack_class is not None) and (not args.check_robustness)

    # multi-gpu training
    # 多 GPU 训练
    if num_gpus > 1: # 若 GPU 数量大于 1
        model = torch.nn.DataParallel(model) #使用 torch 内置的多 GPU 方式
        logger.info("Using torch.nn.DataParallel.")
    logger.info(f"Training model across {num_gpus} GPUs")

    num_train_optimization_steps = (
        int(train_examples_len / args.batch_size / args.grad_accum_steps)
        * args.num_train_epochs
    ) # 计算需要更新的步数

    # 根据不同的模型，设置需要更新的参数与对应的优化器
    if args.model == "lstm" or args.model == "cnn":
    # 如果是 lstm 模型或者 cnn 模型
        def need_grad(x):
            return x.requires_grad
        # 定义判断是否需要梯度更新的函数，使用 filter 对模型参数进行过滤， filter 函数返回判断函数结果为 true 的 iterator
        optimizer = torch.optim.Adam( # 设置 Adam 优化器
            filter(need_grad, model.parameters()), lr=args.learning_rate
        )
        scheduler = None # 没有学习率的更新
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
        wandb = textattack.shared.utils.LazyLoader("wandb", globals(), "wandb")

        wandb.init(sync_tensorboard=True)

    # Save original args to file
    # 将训练参数保存到 json 文件
    args_save_path = os.path.join(args.output_dir, "train_args.json")
    _save_args(args, args_save_path)
    logger.info(f"Wrote original training args to {args_save_path}.")

    tb_writer.add_hparams(
        {k: v for k, v in vars(args).items() if _is_writable_type(v)}, {}
    ) # 将所有可以保存的参数添加到 writer

    # Start training
    # 开始训练
    logger.info("***** Running training *****")
    if augmenter:
        logger.info(f"\tNum original examples = {train_examples_len}")
        logger.info(f"\tNum examples after augmentation = {len(train_text)}")
    else:
        logger.info(f"\tNum examples = {train_examples_len}")
    logger.info(f"\tBatch size = {args.batch_size}")
    logger.info(f"\tMax sequence length = {args.max_length}")
    logger.info(f"\tNum steps = {num_train_optimization_steps}")
    logger.info(f"\tNum epochs = {args.num_train_epochs}")
    logger.info(f"\tLearning rate = {args.learning_rate}")

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

    # 改变 loss 的计算
    def loss_backward(loss):
        # 如果是多 GPU 训练
        if num_gpus > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training， 获取多个 gpu 上训练的平均损失
        if args.grad_accum_steps > 1: # 如果设置了梯度累加步
            loss = loss / args.grad_accum_steps # 对累加后的梯度还原
        loss.backward() # 反传 loss
        return loss

    # 如果标签不是整数，则使用回归损失
    if args.do_regression:
        # TODO integrate with textattack `metrics` package
        loss_fct = torch.nn.MSELoss()  # 最小二乘损失
    else:
        loss_fct = torch.nn.CrossEntropyLoss() # 交叉熵损失

    if args.adversarial_training or args.mixup_training:
        # 仅在设置的对抗训练周期中使用对抗训练样本
        logger.info("Read perturbed dataset from file...")
        dataset = PerturbedDataset(args.file_paths, tokenizer)
        adv_dataloader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size)
        train_dataloader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size)

    # 进行每轮迭代
    for epoch in tqdm.trange(
        int(args.num_train_epochs), desc="Epoch", position=0, leave=True
    ):
        if args.mixup_training:
            # mixup training
            prog_bar = tqdm.tqdm(adv_dataloader, desc="Iteration", position=0, leave=True)  # 进度条

            # Use these variables to track training accuracy during classification.
            correct_predictions = 0  # 正确分类的对抗样本
            total_predictions = 0  # 总共正确分类的对抗样本
            # 对抗训练
            for step, batch in enumerate(zip(prog_bar, train_dataloader)):
                adv_batch, normal_batch = batch
                origin_ids_a, perturb_ids, labels_a = adv_batch
                origin_ids_b, _, labels_b = normal_batch
                labels_a = labels_a.to(device)
                labels_b = labels_b.to(device)

                # 正常输入
                origin_ids_a = origin_ids_a.to(device)
                origin_ids_b = origin_ids_b.to(device)
                perturb_ids = perturb_ids.to(device)
                logits = model(origin_ids_a)  # 获取普通样本的 logits
                logits_normal_mix = model(origin_ids_a, origin_ids_b)
                logits_mix = model(origin_ids_a, perturb_ids)  # 获取 mixup 样本的 logits

                # 计算损失, 使用交叉生损失
                m = np.float32(np.random.beta(1, 1))
                loss = _mixup_criterion(loss_fct, logits_normal_mix, labels_a, labels_b, m)

                prob_logits, prob_mix = F.softmax(logits, dim=1), F.softmax(logits_mix, dim=1)

                p_mixture = torch.clamp((prob_logits + prob_mix) / 2., 1e-7, 1).log()
                loss += 6 * (F.kl_div(p_mixture, prob_logits, reduction='batchmean') +
                              F.kl_div(p_mixture, prob_mix, reduction='batchmean')) / 2.

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
            # Print training accuracy, if we're tracking it.
            # 记录训练的准确率
            if total_predictions > 0:
                train_acc = correct_predictions / total_predictions
                logger.info(f"Train accuracy: {train_acc * 100}%")
                tb_writer.add_scalar("epoch_train_score", train_acc, epoch)
        else:
            # 普通的 training 或是 adversarial training
            prog_bar = tqdm.tqdm(train_dataloader, desc="Iteration", position=0, leave=True)  # 进度条

            # Use these variables to track training accuracy during classification.
            correct_predictions = 0 # 正确分类的对抗样本
            total_predictions = 0 # 总共正确分类的对抗样本
            # 训练
            for step, batch in enumerate(prog_bar):
                input_ids, labels = batch
                labels = labels.to(device)
                if isinstance(input_ids, dict):
                    # 如果传入的 input_ids 是个字典，将字典转为 huggingface 模型需要的输入形状
                    ## dataloader collates dict backwards. This is a workaround to get
                    # ids in the right shape for HuggingFace models
                    input_ids = {
                        k: torch.stack(v).T.to(device) for k, v in input_ids.items()
                    }
                    logits = model(**input_ids)[0]
                else:
                    # 正常输入
                    input_ids = input_ids.to(device)
                    logits = model(input_ids) # 获取模型输出

                # 计算损失
                # 如果开启回归
                if args.do_regression:
                    # TODO integrate with textattack `metrics` package
                    # 使用最小二乘损失
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    # 使用交叉生损失
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
            # Print training accuracy, if we're tracking it.
            # 记录训练的准确率
            if total_predictions > 0:
                train_acc = correct_predictions / total_predictions
                logger.info(f"Train accuracy: {train_acc*100}%")
                tb_writer.add_scalar("epoch_train_score", train_acc, epoch)

        # Check accuracy after each epoch.
        # skip args.num_clean_epochs during adversarial training
        # 如果为正常训练，或对抗训练中使用了对抗样本的轮次，则进行准确性验证
        if (not adversarial_training) or (epoch >= args.num_clean_epochs):
            eval_score = _get_eval_score(model, eval_dataloader, args.do_regression)
            tb_writer.add_scalar("epoch_eval_score", eval_score, epoch)

            # 如果每轮都要保存模型
            if args.checkpoint_every_epoch:
                _save_model_checkpoint(model, args.output_dir, args.global_step)

            # 输出日志
            logger.info(
                f"Eval {'pearson correlation' if args.do_regression else 'accuracy'}: {eval_score*100}%"
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
        # 验证训练后的模型在作为被攻击模型时的健壮性
        if args.check_robustness:
            samples_to_attack = list(zip(eval_text, eval_labels))
            samples_to_attack = random.sample(samples_to_attack, 1000)
            # 使用这些样本对模型生成对抗样本，对模型进行攻击
            adv_attack_results = _generate_adversarial_examples(  # 直接返回攻击后的结果，神奇的函数
                model_wrapper, attack_class, samples_to_attack
            )
            attack_types = [r.__class__.__name__ for r in adv_attack_results]
            attack_types = collections.Counter(attack_types)

            # 对抗训练模型在普通样本上的准确率
            adv_acc = 1 - (
                attack_types["SkippedAttackResult"] / len(adv_attack_results)   # skipped 样本比例，对于模型原本就错分的样本，忽略攻击效果，因此标记为 skipped
            )
            # 总的攻击数为成功与失败的样本数
            total_attacks = (
                attack_types["SuccessfulAttackResult"]
                + attack_types["FailedAttackResult"]
            )
            # 攻击的成功率
            adv_succ_rate = attack_types["SuccessfulAttackResult"] / total_attacks
            # 在对抗样本上的准确性
            after_attack_acc = attack_types["FailedAttackResult"] / len(
                adv_attack_results
            )

            # 记录日志
            tb_writer.add_scalar("robustness_test_acc", adv_acc, global_step)  # 在普通样本上的准确率
            tb_writer.add_scalar("robustness_total_attacks", total_attacks, global_step)  # 进行的攻击次数
            tb_writer.add_scalar(
                "robustness_attack_succ_rate", adv_succ_rate, global_step  # 攻击的成功率
            )
            tb_writer.add_scalar(
                "robustness_after_attack_acc", after_attack_acc, global_step  # 对抗训练后在对抗样本上的准确率
            )

            logger.info(f"Eval after-attacks accuracy: {100*after_attack_acc}%")  # 打印对抗训练后在对抗样本上的准确率

    # read the saved model and report its eval performance
    # 结束所有训练后，载入保存的模型，进行测试
    logger.info("Finished training. Re-loading and evaluating model from disk.")
    model_wrapper = model_from_args(args, args.num_labels)  # model_wrappper 不是从保存的 json 读出来的，而是直接读传入的 args
    model = model_wrapper.model
    model.load_state_dict(torch.load(os.path.join(args.output_dir, args.weights_name)))  # 载入模型
    eval_score = _get_eval_score(model, eval_dataloader, args.do_regression)  # 在正常测试数据上进行评价
    # 记录测试样本上的准确率
    logger.info(
        f"Saved model {'pearson correlation' if args.do_regression else 'accuracy'}: {eval_score*100}%"
    )

    # 如果保存最后一次训练的模型
    if args.save_last:
        _save_model(model, args.output_dir, args.weights_name, args.config_name)

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
