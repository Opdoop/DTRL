import sys
from args_helper import arg_wrapper, run, _path
sys.path.append(_path())

if __name__ == "__main__":
    args = arg_wrapper()  # 获取默认需要的全部参数，还没有详细筛选哪些实际有用，TODO，删除无用参数
    # 根据具体实验设置相应参数
    # 模型类型
    args.model = '../../../bert-base-uncased'
    args.model_type = 'vibert'
    # 数据集
    args.dataset = 'mr'
    args.train_path = '../data/mr/train.txt'
    args.eval_path = '../data/mr/test.txt'

    # lstm/cnn 与 bert 的学习率和 batch size 设置差别蛮大的，还是手动调一下的好
    # args.learning_rate = 2e-2
    args.batch_size = 128
    # args.early_stopping_epochs = 25
    args.num_train_epochs = 15
    args.save_last = False

    # 实验名称
    args.experiment_name = f"{args.model_type}-{args.dataset}-{args.learning_rate}-{args.num_train_epochs}-normal"

    # 跑起来~
    run(args)  # 参数传入主函数