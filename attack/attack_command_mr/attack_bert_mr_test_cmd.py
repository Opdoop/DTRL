import sys
from args_helper import arg_wrapper, _path, model_path
sys.path.append(_path())
from textattack.commands.attack.attack_command import AttackCommand


if __name__ == "__main__":
    args = arg_wrapper()  # 编译参数
    # 从 py 载入数据
    args.dataset_from_file = '../data/mr_test.py'
    # 攻击策略
    # args.recipe = 'pwws'
    print(args.recipe)
    # 攻击的模型地址
    model_name = 'bert-base-uncased-mr-normal'
    args.model = model_path(model_name)
    # 保存结果文件
    args.log_to_csv = '{}/{}-test.csv'.format(model_name, args.recipe)
    attacker = AttackCommand()
    args.parallel = False
    attacker.run(args)  # 参数传入主函数
