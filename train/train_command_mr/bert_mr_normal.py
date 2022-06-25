import sys
from args_helper import arg_wrapper, run, _path
sys.path.append(_path())

if __name__ == "__main__":
    args = arg_wrapper()  # 获取默认需要的全部参数，还没有详细筛选哪些实际有用，TODO，删除无用参数
    # 根据具体实验设置相应参数
    # 模型类型
    args.model = '../../../bert-base-uncased'
    # 数据集
    args.dataset = 'mr'
    args.train_path = '../data/mr/train.txt'
    args.eval_path = '../data/mr/test.txt'

    # lstm/cnn 与 bert 的学习率和 batch size 设置差别蛮大的，还是手动调一下的好
    args.learning_rate = 5e-5
    args.batch_size = 128
    args.early_stopping_epochs = 3
    args.num_train_epochs = 3
    args.model_type = 'None'

    # 对抗训练的相关设置
    args.adversarial_training = False  # 是否进行 adversarial training ，如果为 True ，则读取 file_path 对应的对抗样本
    # 对抗样本的路径
    args.mixup_training = False  # 使用使用 mixup 进行 adversarial training， TODO 确认 MixAug 是怎么做的
    args.mix_normal_example = False
    args.save_last = False

    # 实验名称
    args.experiment_name = f"{args.model}-{args.dataset}-normal"

    # 跑起来~
    run(args)  # 参数传入主函数