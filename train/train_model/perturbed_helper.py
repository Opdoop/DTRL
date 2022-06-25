import csv
from .train_args_helpers import clean_str
from torch.utils.data import Dataset
import numpy as np

def batch_encode(tokenizer, text_list):
    # 对输入的文本进行分词
    if hasattr(tokenizer, "batch_encode"):
        # 如果有 batch_encode 方法
        return tokenizer.batch_encode(text_list) # 推测是返回一个 iterator
    else:
        # 如果分词器没用 batch encode 的方法
        return [tokenizer.encode(text_input) for text_input in text_list] # 对输入的 text list 全部进行分词并转换为字典下标
    # encode 方法应该还包含了添加特殊 token 的步骤

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

class PerturbedDataset(Dataset):
    def __init__(self, file_paths, tokenizer):
        self.tokenizer = tokenizer
        self.file_paths = file_paths
        self.text_list, self.perturbed_list, self.label_list = self.perturbed_dataset()

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, idx):
        return self.text_list[idx], self.perturbed_list[idx], self.label_list[idx]

    def _text_formation(self, text):
        if text.find(">>>>") > 0 :
            premise, hypothesis = text.split(">>>>")
            premise = premise[9:]
            hypothesis = hypothesis[12:]
            text = (premise, hypothesis)
        return text

    def read_csv(self, file_path, clean=False):
        '''
        读取 attacks 生成的 result csv，返回 4 个 list: ground_truth_output, original_text, perturbed_text, result_type
        result csv 的列头分别为：
        "ground_truth_output","num_queries","original_output","original_score","original_text",\
        "perturbed_output","perturbed_score","perturbed_text","result_type"
        :param file_path: 路径文件
        :return: 返回 list 结果
        '''
        with open(file_path, 'r', encoding='utf8') as fin:
            reader = csv.reader(fin)
            next(reader) # 跳过表头
            ground_list, original_list, perturbed_list, result_list = [], [], [], []
            for line in reader:
                ground_truth_output, _, _, _, original_text, _, _, perturbed_text, result_type = line
                if clean:
                    original_text = clean_str(original_text)
                    perturbed_text = clean_str(perturbed_text)
                ground_list.append(float(ground_truth_output))
                original_list.append(self._text_formation(original_text))
                perturbed_list.append(self._text_formation(perturbed_text))
                result_list.append(result_type)
        return ground_list, original_list, perturbed_list, result_list

    def read_perturbed_text(self, file_path):
        '''
        拼接多个 attack_result 的结果，返回 dataset
        # TO-DO 输入至少 1 个文件路径，返回 dataloader 需要输入的 dataset (orginal_text, perturbed_text_1, perturbed_text_2, ...), ground_truth_label
        :param file_paths: 文件地址 list
        :return: dataset
        '''
        ground_list, original_list, perturbed_list, result_list = self.read_csv(file_path)
        orginal_idx = _batch_encoder(self.tokenizer, original_list)  # tokenizer 将 text 转为 idx
        perturbed_idx = _batch_encoder(self.tokenizer, perturbed_list) # tokenizer 将 text 转为 idx

        text_list, perturbed_list, label_list, idx_list = [], [], [], []
        for idx, result in enumerate(result_list):
            if result == 'Successful':  # 获取攻击类型为成功的样本
                text_list.append(orginal_idx[idx])
                perturbed_list.append(perturbed_idx[idx])
                label_list.append(int(ground_list[idx]))
                idx_list.append(idx)
        text_list = np.array(text_list)  # 输入的字典 id 转化为 np 数组
        perturbed_list = np.array(perturbed_list)  # 输入的字典 id 转化为 np 数组
        label_list = np.array(label_list)  # 对于标签也转换为 np 数组
        return text_list, perturbed_list, label_list

    def perturbed_dataset(self):
        '''
        对于攻击结果文件有多个的情况，所有结果拼接在一起，tokenize 之后的结果
        :return:
        '''
        text_all, perturbed_all, label_all = [], [], []
        for path in self.file_paths:
            text_list, perturbed_list, label_list = self.read_perturbed_text(path)
            text_all.extend(text_list)
            perturbed_all.extend(perturbed_list)
            label_all.extend(label_list)
        return text_all, perturbed_all, label_all

    def perturbed_string(self):
        '''
        返回 string
        :return:
        '''
        perturbed_all, label_all = [], []
        for path in self.file_paths:
            ground_list, original_list, perturbed_list, result_list = self.read_csv(path)
            for i in range(len(result_list)):
                result = result_list[i]
                if result == 'Successful':
                    perturbed_all.append(perturbed_list[i])
                    label_all.append(int(ground_list[i]))
        return perturbed_all, label_all
