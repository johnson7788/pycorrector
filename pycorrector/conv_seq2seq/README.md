# Neural Text Error Correction with Conv Seq2Seq Model

## 特点

该模型具有以下特点：

- ```Attention based seq2seq framework.```
编码器和解码器可以是LSTM或GRU。可以使用三种不同的对齐方法来计算注意力得分。

- ```CONV seq2seq network.```

- ```Beam search algorithm.```
我们实现了一种有效的beam search算法，该算法还可以处理batch_size> 1时的情况。

- ```Unknown words replacement.```
该meta-algorithm 可以与任何基于注意力的seq2seq模型一起使用。
OOV单词UNK使用注意力权重手动替换为源文章中的单词。
This meta-algorithm can be used along with any attention based seq2seq model.

## 使用方式

### 依赖
* pip安装依赖包
```bash
pip install fairseq==0.9.0 torch==1.5.0

# 安装python3.7 
pip install fairseq==0.10 torch==1.5.0 pypinyin six
```


### Preprocess

- mini数据集
```
cd conv_seq2seq
python preprocess.py
```
在本目录下生成output, 先生成原始文件
train.src
train.trg
valid.src
valid.trg
然后调用fairseq, 生成的bin目录下文件
调用fairseq，生成按字符拆分的数据, train data(`train.src` and `train.trg`), valid data(`valid.src` and `valid.trg`)

result:
```
# train.src:
吸 烟 对 人 的 健 康 有 害 处 ， 这 是 各 个 人 都 知 道 的 事 实 。
也 许 是 个 家 庭 都 有 子 女 而 担 心 子 女 的 现 在 以 及 未 来 。
如 服 装 ， 若 有 一 个 很 流 行 的 形 式 ， 人 们 就 赶 快 地 追 求 。

# train.trg:
吸 烟 对 人 的 健 康 有 害 处 ， 这 是 每 个 人 都 知 道 的 事 实 。
也 许 每 个 家 庭 都 有 子 女 而 担 心 子 女 的 现 在 和 未 来 。
如 服 装 ， 若 有 一 个 很 流 行 的 样 式 ， 人 们 就 赶 快 地 追 求 。
```
bin目录下文件需要存在，才能进行训练，是根据output下的各种src和trg文件生成的, 调用fairseq生成的

- 方法二：下载大数据集

1. download from https://pan.baidu.com/s/1BkDru60nQXaDVLRSr7ktfA  密码:m6fg [130W sentence pair，215MB], put data to `conv_seq2seq/output` folder.
2. run `preprocess.py`.
```
python preprocess.py
```

generate fairseq format data to `bin` folder:
```
> tree conv_seq2seq/output
conv_seq2seq/output
├── bin
│   ├── dict.src.txt
│   ├── dict.trg.txt
│   ├── train.src-trg.src.bin
│   ├── train.src-trg.src.idx
│   ├── train.src-trg.trg.bin
│   ├── train.src-trg.trg.idx
│   ├── valid.src-trg.src.bin
│   ├── valid.src-trg.src.idx
│   ├── valid.src-trg.trg.bin
│   └── valid.src-trg.trg.idx
├── train.src
├── train.trg
├── valid.src
└── valid.trg
```

### 训练

```
sh train.sh
```

### Infer
修改config.py中训练好的模型的路径，开始推理
```
python infer.py

```

### Result
```
input: 少先队员因该给老人让坐 output: 少先队员因该给老人让座
input: 少先队员应该给老人让坐 output: 少先队员应该给老人让座
input: 没有解决这个问题， output: 没有解决这个问题，，
input: 由我起开始做。 output: 由我起开始做
input: 由我起开始做 output: 由我开始做

```

## Reference
1. [《基于深度学习的中文文本自动校对研究与实现》[杨宗霖, 2019]](https://github.com/shibing624/pycorrector/blob/master/docs/基于深度学习的中文文本自动校对研究与实现.pdf)
2. [《A Sequence to Sequence Learning for Chinese Grammatical Error Correction》[Hongkai Ren, 2018]](https://link.springer.com/chapter/10.1007/978-3-319-99501-4_36)
2. [《Neural Abstractive Text Summarization with Sequence-to-Sequence Models》[Tian Shi, 2018]](https://arxiv.org/abs/1812.02303)
