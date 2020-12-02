import sys
sys.path.append('../..')
from sklearn.model_selection import train_test_split
from pycorrector.utils.tokenizer import segment
from pycorrector.conv_seq2seq import config


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

    # 用空格分隔
    for src_line, trg_line in zip(src_lines, trg_lines):
        source = segment(src_line.strip(), cut_type='char')
        target = segment(trg_line.strip(), cut_type='char')
        data_pairs.append((source, target))
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


def gen_fairseq_data(source_lang, target_lang, trainpref, validpref, nwordssrc, nwordstgt, destdir):
    """
    :param source_lang: 源语言，源句子
    :param target_lang: 目标语言，改正后的句子
    :param trainpref:
    :param validpref:
    :param nwordssrc:  源语言的最大单词数
    :param nwordstgt: 目标语言的最大单词数
    :param destdir:
    :return:
    """
    from fairseq import options
    from fairseq_cli import preprocess

    parser = options.get_preprocessing_parser()
    args = parser.parse_args()

    args.source_lang = source_lang
    args.target_lang = target_lang
    args.trainpref = trainpref
    args.validpref = validpref
    args.nwordssrc = nwordssrc
    args.nwordstgt = nwordstgt
    args.destdir = destdir
    preprocess.main(args)
    print(f"处理源文件成功，生成文件到{destdir}目录下，包括字典文件和训练文件")

if __name__ == '__main__':
    # data_list = read_file_split(srcfile="data/mydata/train.src", trgfile="data/mydata/train.trg")
    # save_corpus_data(data_list, "output/train", "output/test", "output/valid")
    gen_fairseq_data(source_lang="src",
                     target_lang="trg",
                     trainpref=config.trainpref,
                     validpref=config.valpref,
                     nwordssrc=config.vocab_max_size,
                     nwordstgt=config.vocab_max_size,
                     destdir=config.data_bin_dir)
