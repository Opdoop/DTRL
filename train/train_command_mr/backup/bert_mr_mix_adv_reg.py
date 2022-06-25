import sys
from args_helper import arg_wrapper, run, _path, _adv_path
sys.path.append(_path())

if __name__ == "__main__":
    args = arg_wrapper()  # 获取默认需要的全部参数，还没有详细筛选哪些实际有用，TODO，删除无用参数
    # 根据具体实验设置相应参数
    # 模型类型
    args.model = 'bert-base-uncased'
    # 数据集
    args.dataset = 'mr'
    args.train_path = '../data/mr/train.txt'
    args.eval_path = '../data/mr/test.txt'

    # lstm/cnn 与 bert 的学习率和 batch size 设置差别蛮大的，还是手动调一下的好
    args.learning_rate = 5e-5
    args.batch_size = 32
    args.early_stopping_epochs = 5
    args.num_train_epochs = 5

    # 对抗训练的相关设置
    args.adversarial_training = False  # TODO
    args.mixup_training = True  # 使用使用 mixup 进行 adversarial training， TODO
    args.mix_normal_example = False    # TODO
    args.regularized_adv_example = True   # TODO
    args.adv_mixup = False #

    # 对抗样本的路径
    victim_model = f"{args.model}-{args.dataset}-normal"
    recipe = 'multi'
    args.file_paths = [_adv_path(victim_model, 'deepwordbug-train.csv'),
                       _adv_path(victim_model, 'textfooler-train.csv'),
                       _adv_path(victim_model, 'textbugger-train.csv')]

    # 实验名称
    args.experiment_name = f"{args.model}-{args.dataset}-mix-adv-reg-{recipe}"  # TODO
    args.mix_weight = 8  # 默认为 6

    # 跑起来~
    run(args)  # 参数传入主函数