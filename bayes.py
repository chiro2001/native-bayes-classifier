import os
import re
import numpy as np
from functools import reduce
import random


def split_text(text: str):
    # 将特殊符号作为切分标志进行字符串切分，即非字母、非数字
    tokens = re.split(r'\W+', text)
    # 除了单个字母，例如大写的I，其它单词变成小写；并且排除纯数字
    return [tok.lower() for tok in tokens if len(tok) > 2 and not tok.isdigit()]


def get_words_data():
    """
    获取训练用文本数据，并且格式化分割到单词
    :return: 得到的文本数据, 分类列表
    """
    class_types = [r for r in os.listdir('email') if os.path.isdir(os.path.join('email', r))]

    def read_data(filename_: str) -> str:
        with open(filename_, 'r', encoding='gbk') as f:
            return f.read()

    words_data = []
    for c in class_types:
        for filename in os.listdir(os.path.join('email', c)):
            file_data = read_data(os.path.join(f'email/{c}', filename))
            words_data.append((c, split_text(file_data)))
    return words_data, class_types


def get_words_label(words_data: list) -> list:
    """
    得到当前数据集下的词汇表
    :param words_data: 读取到的词语数据
    :return: 词汇表
    """
    # 使用 set 去重
    words_label = set({})
    for words in words_data:
        words_label.update(words[1])
    res = list(words_label)
    res.sort()
    return res


def get_words_vector(words, words_label: list) -> list:
    """
    得到一个词语或者一个词语列表对应的词向量
    :param words: 词语或者词语列表
    :param words_label: 词汇表
    :return: 词向量
    """
    return [(1 if val == words else 0) if not isinstance(words, list)
            else (1 if val in words else 0) for val in words_label]


def native_bayes_train(words_label: list, train_data: list, class_types: list, alpha: float = 1e-3):
    """
    训练朴素贝叶斯分类器
    :param words_label: 词汇表
    :param train_data: 训练数据集
    :param class_types: 分类列表
    :param alpha: > 0 且很小的数，用于拉普拉斯平滑
    :return: 训练结果, 各种类的概率
    """
    # 初始化就加上这个 alpha，防止出现 Nan。
    p_result = {c: np.array([alpha for _ in range(len(words_label))]) for c in class_types}
    p_b = {c: alpha for c in class_types}
    for data in train_data:
        # 得到词向量
        words_vector = np.array(get_words_vector(data[1], words_label))
        # 记录到训练数据
        p_result[data[0]] += words_vector + alpha
        # 记录到对应分类的概率
        p_b[data[0]] += 1 / len(train_data)
    for k in p_result:
        p_result[k] = np.log(p_result[k])
    return p_result, p_b


def native_bayes_test(words_label: list, p_result: dict, test_data: list, p_b: dict, alpha: float = 1e-3) -> float:
    """
    测试朴素贝叶斯分类器
    :param words_label: 词汇表
    :param p_result: 训练结果
    :param test_data: 测试数据集
    :param p_b: 各种类的概率
    :param alpha: > 0 且很小的数，用于拉普拉斯平滑
    :return: 测试正确的概率
    """
    accurate = 0
    for data in test_data:
        # 获得词向量
        words_vector = np.array(get_words_vector(data[1], words_label))
        # 统计相对概率
        Qs = {key: np.log(p_b[key]) + sum(p_result[key] * words_vector) - np.log(sum(words_vector + alpha)) for key in
              p_result}
        # 选取相对概率最高的那个作为分类结果
        classification = reduce((lambda x, y: x if x[1] > y[1] else y), ((k, Qs[k]) for k in Qs))[0]
        print(f"result {classification == data[0]}, classification = {classification}, label = {data[0]}; Qs: {Qs}")
        # 记录正确率
        accurate += (1 if classification == data[0] else 0) / len(test_data)
    return accurate


def main():
    # 读取训练、测试数据
    words_data, class_types = get_words_data()
    # 打乱训练数据
    random.shuffle(words_data)
    # 生成词汇表
    words_label = get_words_label(words_data)
    # 训练朴素贝叶斯分类器，使用随机40份邮件做训练数据集
    p_result, p_b = native_bayes_train(words_label, words_data[:40], class_types)
    # 测试朴素贝叶斯分类器，使用剩下的邮件做测试数据集
    accurate = native_bayes_test(words_label, p_result, words_data[40:], p_b)
    print(f"DONE! accurate = {(accurate * 100):.2f}%")


if __name__ == '__main__':
    main()
