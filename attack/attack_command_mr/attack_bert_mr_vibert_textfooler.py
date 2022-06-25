import sys
from args_helper import arg_wrapper, _path, model_path
sys.path.append(_path())
from textattack.commands.attack.attack_command import AttackCommand


if __name__ == "__main__":
    args = arg_wrapper()  # 编译参数
    # 从 py 载入数据
    args.dataset_from_file = '../data/mr_test.py'
    # 攻击策略
    args.recipe = 'textfooler'
    # 攻击的模型地址
    model_name = f'vibert'
    args.model_from_file = './models/{}.py'.format(args.file_path)
    args.parallel = False #True
    # 保存结果文件
    import pdb
    pdb.set_trace()
    args.log_to_csv = '{}/{}-{}.csv'.format(model_name, args.file_path, args.recipe)
    attacker = AttackCommand()
    attacker.run(args)  # 参数传入主函数
