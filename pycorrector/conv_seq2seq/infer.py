# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys

from fairseq import options

sys.path.append('../..')

from pycorrector.conv_seq2seq import config
from pycorrector.conv_seq2seq import interactive


def infer(model_path, vocab_dir, arch, test_data, max_len, temperature):
    parser = options.get_generation_parser(interactive=True)
    parser.set_defaults(arch=arch,
                        input=test_data,
                        max_tokens=max_len,
                        temperature=temperature,
                        path=model_path)
    args = options.parse_args_and_arch(parser, input_args=[vocab_dir])
    args.max_sentences = 1
    return interactive.main(args)


def infer_interactive(model_path, vocab_dir, arch, max_len, temperature=1.0):
    return infer(model_path, vocab_dir, arch, '-', max_len, temperature)


if __name__ == '__main__':
    # 通过文本预测
    inputs = [
        '6 结 论 与 未 来 工 作',
        '我 们 逐 层 初 始 化 S A E 网 络 ， 每 一 层 都 是 经 过 降 噪 处 理 的 自 动 编 码 器 ， 经 过 训 练 ， 可 以 在 随 机 破 坏 后 重 建 上 一 层 的 输 出 （ V i n c e n t 等 ， 2 0 1 0 ） 。 去 噪 自 动 编 码 器 是 两 层 神 经 网 络 ， 定 义 为 ：',
        '我 们 的 培 训 策 略 可 以 看 作 是 一 种 自 我 培 训 （ N i g a m ＆ G h a n i ， 2 0 0 0 ） 。 与 自 训 练 一 样 ， 我 们 采 用 初 始 分 类 器 和 未 标 记 的 数 据 集 ， 然 后 使 用 分 类 器 标 记 数 据 集 ， 以 便 对 其 自 身 的 高 置 信 度 预 测 进 行 训 练 。 确 实 ， 在 实 验 中 我 们 观 察 到 ， 通 过 从 高 置 信 度 预 测 中 学 习 ， D E C 可 以 在 每 次 迭 代 中 提 高 初 始 估 计 值 ， 从 而 有 助 于 改 善 低 置 信 度 预 测 。',
        '优 化 D E C 具 有 挑 战 性 。 我 们 要 同 时 解 决 集 群 分 配 和 基 础 特 征 表 示 。 但 是 ， 与 监 督 学 习 不 同 ， 我 们 不 能 使 用 标 记 数 据 来 训 练 我 们 的 深 度 网 络 。 相 反 ， 我 们 建 议 使 用 从 当 前 软 集 群 分 配 派 生 的 辅 助 目 标 分 布 来 迭 代 地 精 炼 集 群 。 此 过 程 逐 渐 改 善 了 聚 类 以 及 特 征 表 示 。',
        '不能人类实现更美好的将来。',
        '这几年前时间，',
        '歌曲使人的感到快乐，',
        '少先队员因该为老人让坐，',
        '会能够大幅减少互相抱怨的情况。'
    ]
    outputs = infer(model_path=config.best_model_path,
                    vocab_dir=config.data_bin_dir,
                    arch=config.arch,
                    test_data=[' '.join(list(i)) for i in inputs],
                    max_len=config.max_len,
                    temperature=config.temperature)
    print("output:", outputs)

    # 通过文件预测
    outputs = infer(model_path=config.best_model_path,
                    vocab_dir=config.data_bin_dir,
                    arch=config.arch,
                    test_data=config.val_src_path,
                    max_len=config.max_len,
                    temperature=config.temperature)
    print("output:", outputs)
