# -*- coding: utf-8 -*-
# @Time    : 2019/12/27 7:22 下午
# @Author  : lizhen
# @FileName: data_process.py
from config import WeiboConfig
import pandas as pd
from utils import clean_line, split_train_test, tokenizer
from torchtext import data
import torch


def weibo_data_process():
    weibo_config = WeiboConfig()
    weibo_data = pd.read_csv(weibo_config.raw_data)
    # 数据清洗
    weibo_data['clean_text'] = weibo_data['review'].apply(clean_line)
    weibo_data = weibo_data.drop(columns=['review'])

    # 划分训练集、验证集、测试集
    train_val, test = split_train_test(weibo_data, 0.7)
    train, val = split_train_test(train_val, 0.8)
    # 保存
    train.to_csv(weibo_config.train_csv, index=0, encoding="utf8")
    val.to_csv(weibo_config.val_csv, index=0, encoding="utf8")
    test.to_csv(weibo_config.test_csv, index=0, encoding="utf8")

    # 定义 torchtext 的LABEL,TEXT
    # 注意 LABEL对应：LabelField，TEXT 对应：Field
    LABEL = data.LabelField(sequential=False, use_vocab=False, dtype=torch.float)
    TEXT = data.Field(sequential=True, tokenize=tokenizer)

    # 构建Dataset
    train, val = data.TabularDataset.splits(path=weibo_config.data_path, train='weibo_train.csv',
                                            validation='weibo_val.csv',
                                            format='csv', skip_header=True,
                                            fields=[('label', LABEL), ('text', TEXT)])

    test = data.TabularDataset(path=weibo_config.test_csv, format='csv', skip_header=True,
                               fields=[('label', LABEL), ('text', TEXT)])
    # 构建词汇表
    TEXT.build_vocab(train, max_size=6000)
    LABEL.build_vocab(train)

    # 构建迭代器
    train_iterator = data.Iterator(train, batch_size=64, device=weibo_config.device, sort=False,
                                   sort_within_batch=False,
                                   repeat=False)
    val_iterator = data.Iterator(val, batch_size=64, device=weibo_config.device, sort=False, sort_within_batch=False,
                                 repeat=False)
    test_iterator = data.Iterator(test, batch_size=128, device=weibo_config.device, sort=False, sort_within_batch=False,
                                  repeat=False)

    return TEXT,LABEL,train_iterator,val_iterator,test_iterator

if __name__ == '__main__':
    weibo_data_process()
