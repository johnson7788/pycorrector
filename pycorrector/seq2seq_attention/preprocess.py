# -*- coding: utf-8 -*-
# Author: XuMing(xuming624@qq.com)
# Brief:
import sys
from codecs import open
from sklearn.model_selection import train_test_split

sys.path.append('../..')
from pycorrector.utils.tokenizer import segment
from pycorrector.seq2seq_attention import config


def parse_file(path, line_start, line_end):
    """
    :param path:
    :param line_start: 从第几行开始
    :param line_end:  从第几行结束
    :return:
    """
    print('Parse data from %s' % path)
    data_list = []
    with open(filename=path, mode="r", encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[line_start:line_end]:
            line_split = line.split('\t')
            # 我们只使用原始的错误语句和最终的正确句子
            src = line_split[2]
            trg = line_split[-1]

            source = segment(src.strip(), cut_type='char')
            target = segment(trg.strip(), cut_type='char')

            pair = [source, target]
            if pair not in data_list:
                data_list.append(pair)
    return data_list


def _save_data(data_list, data_path):
    print("开始写入文件,默认追加模式")
    with open(data_path, 'a+', encoding='utf-8') as f:
        count = 0
        for src, dst in data_list:
            f.write(' '.join(src) + '\t' + ' '.join(dst) + '\n')
            count += 1
        print("save line size:%d to %s" % (count, data_path))


def _save_data2(data_list, seq1_path, seq2_path):
    """
    写入成到seq1和seq2路径
    :param data_list:
    :param data_path:
    :return:
    """
    print("开始写入文件,默认追加模式")
    with open(seq1_path, 'a+', encoding='utf-8') as f1:
        with open(seq2_path, 'a+', encoding='utf-8') as f2:
            count = 0
            for src, dst in data_list:
                f1.write(' '.join(src) + '\n')
                f2.write(' '.join(dst) + '\n')
                count += 1
            print("save line size:%d to %s %s" % (count, seq1_path, seq1_path))


def save_corpus_data(data_list, train_data_path, test_data_path):
    train_lst, test_lst = train_test_split(data_list, test_size=0.1)
    _save_data(train_lst, train_data_path)
    _save_data(test_lst, test_data_path)


def save_corpus_data2(data_list, train_data_path, test_data_path, valid_data_path):
    train_val_lst, test_lst = train_test_split(data_list, test_size=0.1, random_state=1)
    train_lst, valid_lst = train_test_split(train_val_lst, test_size=0.1, random_state=1)
    _save_data2(train_lst, train_data_path + ".src", train_data_path + ".trg")
    _save_data2(test_lst, test_data_path + ".src", test_data_path + ".trg")
    _save_data2(test_lst, valid_data_path + ".src", valid_data_path + ".trg")


if __name__ == '__main__':
    # train data
    data_list = []
    for path in config.raw_train_paths:
        data_list.extend(parse_file(path, line_start=0, line_end=2001))
    # save_corpus_data(data_list, config.train_path, config.test_path)
    save_corpus_data2(data_list, "output/train", "output/test", "output/valid")
