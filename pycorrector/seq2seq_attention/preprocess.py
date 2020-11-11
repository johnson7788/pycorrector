# -*- coding: utf-8 -*-
# Author: XuMing(xuming624@qq.com)
# Brief:
import sys
from codecs import open
from sklearn.model_selection import train_test_split

sys.path.append('../..')
from pycorrector.utils.tokenizer import segment
from pycorrector.seq2seq_attention import config


def parse_file(path):
    print('Parse data from %s' % path)
    data_list = []
    with open(filename=path, mode="r", encoding='utf-8') as file:
        for line in file:
            line_split = line.split('\t')
            #我们只使用原始的错误语句和最终的正确句子
            src = line_split[2]
            trg = line_split[-1]

            source = segment(src.strip(), cut_type='char')
            target = segment(trg.strip(), cut_type='char')

            pair = [source, target]
            if pair not in data_list:
                data_list.append(pair)
    return data_list

def _save_data(data_list, data_path):
    with open(data_path, 'w', encoding='utf-8') as f:
        count = 0
        for src, dst in data_list:
            f.write(' '.join(src) + '\t' + ' '.join(dst) + '\n')
            count += 1
        print("save line size:%d to %s" % (count, data_path))


def save_corpus_data(data_list, train_data_path, test_data_path):
    train_lst, test_lst = train_test_split(data_list, test_size=0.1)
    _save_data(train_lst, train_data_path)
    _save_data(test_lst, test_data_path)


if __name__ == '__main__':
    # train data
    data_list = []
    for path in config.raw_train_paths:
        data_list.extend(parse_file(path))
    save_corpus_data(data_list, config.train_path, config.test_path)
