# Seq2seq Attention Model


## Features

* 基于Attention机制的sequence to sequence模型
* BahdanauAttention
* 训练加速tricks：dataset bucketing, prefetching, token-based batching, gradients accumulation
* beam search

## Usage

### Requirements
* pip安装依赖包
```
tensorflow>=2.0.0 # tensorflow-gpu>=2.0.0
```
# 修改配置文件config.py，修改原始文件读取的位置
注意函数
```buildoutcfg
def parse_file(path, line_start, line_end):
```

### Preprocess


- 生成训练和测试数据，拆分比例9：1
```
cd seq2seq_attention
python preprocess.py
```

生成文件在目录output/test.txt 和output/train.txt 下，逐个字分隔

例如:
```
# train.txt:
如 服 装 ， 若 有 一 个 很 流 行 的 形 式 ， 人 们 就 赶 快 地 追 求 。\t如 服 装 ， 若 有 一 个 很 流 行 的 样 式 ， 人 们 就 赶 快 地 追 求 。
```

![short train](../../docs/git_image/short_train.png)

- 正式数据集下载, 数据量比较大，
1. download from https://pan.baidu.com/s/1BkDru60nQXaDVLRSr7ktfA  密码:m6fg [130W sentence pair，215MB], put data to `seq2seq_attention/output` folder.
2. run `preprocess.py`.  生成output下的trg和src文件
```
python preprocess.py
```


### Train

```
python train.py
```

### Infer
```
python infer.py

```

### Result
```
input: 少先队员应该给老人让坐 output: 少先队员应该给老人让座
input: 由我起开始做 output: 由我开始做

```
![short correct result](../../docs/git_image/short_result.png)