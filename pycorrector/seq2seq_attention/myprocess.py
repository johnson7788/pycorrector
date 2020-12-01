from sklearn.model_selection import train_test_split
from pycorrector.utils.tokenizer import segment

def read_file_split(srcfile, trgfile):
    """
    读取我自己的src和trg文件，进行空格分隔，拆分出dev，test，和train
    :param srcfile:
    :param trgfile:
    :return:
    """
    # 保存到的文件
    data_pairs = []
    with open(srcfile, "r") as srcf, open(trgfile, "r") as trgf:
        src_lines = srcf.readlines()
        trg_lines = trgf.readlines()

    #用空格分隔
    for src_line, trg_line in zip(src_lines, trg_lines):
        source = segment(src_line.strip(), cut_type='char')
        target = segment(trg_line.strip(), cut_type='char')
        data_pairs.append((source,target))
    print(f"总的样本数{len(data_pairs)}")
    return data_pairs

def _save_data(data_list, seq1_path, seq2_path):
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
            print("写入行数:%d to %s %s" % (count, seq1_path, seq1_path))

def save_corpus_data(data_list, train_data_path, test_data_path, valid_data_path):
    train_val_lst, test_lst = train_test_split(data_list, test_size=0.1, random_state=1)
    train_lst, valid_lst = train_test_split(train_val_lst, test_size=0.1, random_state=1)
    _save_data(train_lst, train_data_path + ".src", train_data_path + ".trg")
    _save_data(test_lst, test_data_path + ".src", test_data_path + ".trg")
    _save_data(test_lst, valid_data_path + ".src", valid_data_path + ".trg")

if __name__ == '__main__':
    data_list = read_file_split(srcfile="data/mydata/train.src", trgfile="data/mydata/train.trg")
    save_corpus_data(data_list, "output/train", "output/test", "output/valid")