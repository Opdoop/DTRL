import sys
from args_helper import arg_wrapper, run, _path
sys.path.extend(_path())

if __name__ == "__main__":
    args = arg_wrapper()  # 获取默认需要的全部参数，还没有详细筛选哪些实际有用，TODO，删除无用参数
    # 根据具体实验设置相应参数
    # 模型类型
    args.model = '../../../bert-base-uncased'
    args.model_type = 'vibert'
    # 数据集
    args.dataset = 'snli'
    args.train_path = '../data/snli/snli_1.0_train.txt'
    args.eval_path = '../data/snli/snli_1.0_test.txt'

    # lstm/cnn 与 bert 的学习率和 batch size 设置差别蛮大的，还是手动调一下的好
    # args.learning_rate = 5e-5
    args.learning_rate = 2e-05
    args.batch_size = 64
    args.early_stopping_epochs = 5
    args.num_train_epochs = 5
    args.save_last = False
    args.beta = 1e-2

    # args.beta = 0

    # 实验名称
    args.experiment_name = f"{args.model_type}-{args.dataset}-{args.learning_rate}-{args.beta}"

    # 跑起来~
    run(args)  # 参数传入主函数