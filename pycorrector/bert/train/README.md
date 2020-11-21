
## Language model 语言模型训练

下载链接 [`run_language_modeling.py`](https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_language_modeling.py).


在GPT，GPT-2，BERT，DistilBERT和RoBERTa的文本数据集上微调(或从头训练)用于语言建模的库模型。 
GPT和GPT-2使用因果语言建模(CLM)损失进行了微调，而BERT，DistilBERT和RoBERTa使用Masked语言建模(MLM)损失进行微调。

在运行下面的样本之前，您应该获得一个包含text格式的文件，
在该文件上将训练或微调语言模型。此类文本的一个很好的例子是 [WikiText-2 dataset](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/).

我们将引用两个不同的文件：`$ TRAIN_FILE`，其中包含用于训练的文本，以及`$ TEST_FILE`，其中包含将用于评估的文本。

### GPT-2/GPT and causal language modeling

下面的样本在WikiText-2上微调GPT-2。我们正在使用原始的WikiText-2(在tokenization之前没有替换任何tokens)。这里的损失是因果语言建模的损失。

```bash
export TRAIN_FILE=/path/to/dataset/wiki.train.raw
export TEST_FILE=/path/to/dataset/wiki.test.raw

python run_language_modeling.py \
    --output_dir=output \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE
```

在单个K80 GPU上训练大约需要半小时，而评估运行大约需要一分钟。在数据集上进行微调后，其perplexity约为20 score。

### RoBERTa/BERT/DistilBERT and masked language modeling

下面的样本在WikiText-2上微调RoBERTa。在这里，我们也使用原始的WikiText-2。
由于BERT / RoBERTa具有双向机制，因此损失有所不同。因此，我们所使用的损失与训练前的损失相同：mask语言建模。

根据RoBERTa的论文，我们使用dynamic masking而不是static masking。因此，模型收敛的速度可能会稍慢(过多epochs会过拟合)。

我们使用`--mlm` flag ，更改其损失函数为mask损失。

```bash
export TRAIN_FILE=/path/to/dataset/wiki.train.raw
export TEST_FILE=/path/to/dataset/wiki.test.raw

python run_language_modeling.py \
    --output_dir=output \
    --model_type=roberta \
    --model_name_or_path=roberta-base \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm
```

### XLNet and permutation language modeling XLNet和排列语言建模

XLNet使用不同的训练目标，即排列语言建模。
这是一种自动回归方法，通过在输入序列factorization排列上最大化expected likelihood 来学习双向上下文。

我们使用`--plm_probability` flag来定义排列语言模型的mask标签范围的长度与周围上下文长度的比率。

`--max_span_length` flag 还可用于限制用于排列语言模型的mask标签的范围的长度。

```bash
export TRAIN_FILE=/path/to/dataset/wiki.train.raw
export TEST_FILE=/path/to/dataset/wiki.test.raw

python run_language_modeling.py \
    --output_dir=output \
    --model_name_or_path=xlnet-base-cased \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
```
