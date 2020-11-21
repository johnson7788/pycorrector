# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: use bert detect and correct chinese char error
"""

import operator
import os
import sys
import time

from transformers import pipeline

sys.path.append('../..')
from pycorrector.utils.text_utils import is_chinese_string, convert_to_unicode
from pycorrector.utils.logger import logger
from pycorrector.corrector import Corrector

pwd_path = os.path.abspath(os.path.dirname(__file__))


class BertCorrector(Corrector):
    def __init__(self, bert_model_dir=os.path.join(pwd_path, '../data/bert_models/chinese_finetuned_lm/')):
        """
        初始化一个纠错推断模型，
        :param bert_model_dir: 模型目录，包含完整的模型文件，配置文件等, 单词表
        """
        super(BertCorrector, self).__init__()
        self.name = 'bert_corrector'
        # 用于计算耗费时间
        t1 = time.time()
        # 初始化一个pipeline
        self.model = pipeline('fill-mask',
                              model=bert_model_dir,
                              tokenizer=bert_model_dir)
        if self.model:
            self.mask = self.model.tokenizer.mask_token
            logger.debug('加载完成bert模型: %s, 耗时: %.3f s.' % (bert_model_dir, time.time() - t1))

    def bert_correct(self, text):
        """
        句子纠错
        :param text: 句子文本, 单个text,  eg: '疝気医院那好 为老人让坐，疝気专科百科问答'
        :return: corrected_text, list[list], [error_word, correct_word, begin_pos, end_pos]
        """
        #用于保存生成的新的text
        text_new = ''
        details = []
        #加载常用字，# 同音字,# 形似字
        self.check_corrector_initialized()
        # 编码统一，utf-8 to unicode
        text = convert_to_unicode(text)
        # 长句切分为短句, eg: [('疝気医院那好', 0), (' ', 6), ('为老人让坐', 7), ('，', 12), ('疝気专科百科问答', 13)]
        blocks = self.split_2_short_text(text, include_symbol=True)
        for blk, start_idx in blocks:
            # blk代表每个短句子, 组成的新句子
            blk_new = ''
            for idx, s in enumerate(blk):
                # 处理中文错误, 迭代每个字，eg: s代表一个字：'疝', 逐个中文字Mask掉然后预测
                if is_chinese_string(s):
                    #如果这个字符是汉字, sentence_lst eg: ['疝', '気', '医', '院', '那', '好']
                    sentence_lst = list(blk_new + blk[idx:])
                    #把这个句子的idx改成MASK， eg: ['[MASK]', '気', '医', '院', '那', '好']
                    sentence_lst[idx] = self.mask
                    #然后组成新句子
                    sentence_new = ''.join(sentence_lst)
                    # 预测，默认取top5, 预测这个被MASK的token, 返回形式是 eg: 返回5个, 每个类似, 这里假设预测第一个Mask的字, 第一{'sequence': '[CLS] 人 気 医 院 那 好 [SEP]', 'score': 0.040772877633571625, 'token': 782, 'token_str': '人'}
                    predicts = self.model(sentence_new)
                    top_tokens = []
                    for p in predicts:
                        # 提取token的id，如果不存在，给出默认值0，不会不存在的
                        token_id = p.get('token', 0)
                        token_str = self.model.tokenizer.convert_ids_to_tokens(token_id)
                        # top_tokens 是最可能的那五个字： ['人', '哪', '这', '地', '五']
                        top_tokens.append(token_str)
                    #如果top_tokens存在，并且预测出来的字和被mask掉的本身的字不相同，那么进行下一步筛选
                    if top_tokens and (s not in top_tokens):
                        # 取得所有可能正确的词, 根据自定义的形近字和形音字和自定义，取出可能的候选词
                        candidates = self.generate_items(s)
                        if candidates:
                            for token_str in top_tokens:
                                if token_str in candidates:
                                    #如果bert模型预测出来的字在候选的字里面，那么久加入details中
                                    details.append([s, token_str, start_idx + idx, start_idx + idx + 1])
                                    s = token_str
                                    break
                blk_new += s
            # 把每个句子，组成为整个段落
            text_new += blk_new
        details = sorted(details, key=operator.itemgetter(2))
        return text_new, details


if __name__ == "__main__":
    d = BertCorrector()
    error_sentences = [
        '疝気医院那好 为老人让坐，疝気专科百科问答',
        '少先队员因该为老人让坐',
        '少 先  队 员 因 该 为 老人让坐',
        '机七学习是人工智能领遇最能体现智能的一个分知',
        '今天心情很好',
    ]
    for sent in error_sentences:
        corrected_sent, err = d.bert_correct(sent)
        print("original sentence:{} => {}, err:{}".format(sent, corrected_sent, err))
