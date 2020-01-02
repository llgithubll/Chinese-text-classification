# -*- coding: utf-8 -*-
# @Time    : 2019/12/27 7:22 下午
# @Author  : lizhen
# @FileName: data_process.py
from config import WeiboConfig, CnewsConfig,BertConfig
import pandas as pd
from utils import clean_line, split_train_test, tokenizer, generate_bigrams
from torchtext import data
from torchtext.vocab import Vectors
import torch
import os
from utils import tokenize_and_cut, bert_tokenizer_init


def weibo_data_process():
    weibo_config = WeiboConfig()
    # 加载数据
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
    # LABEL = data.LabelField(sequential=False, use_vocab=False, dtype=torch.float) # 二分类
    LABEL = data.LabelField(sequential=False, use_vocab=False)
    TEXT = data.Field(sequential=True, tokenize=tokenizer, include_lengths=True)  # include_lengths=True for LSTM
    # TEXT = data.Field(sequential=True, tokenize=tokenizer,preprocessing = generate_bigrams,
    # include_lengths=True) # FastText
    # TEXT = data.Field(sequential=True, tokenize=tokenizer, batch_first=True)  # batch_first = True for TextCNN

    # 构建Dataset
    train, val = data.TabularDataset.splits(path=weibo_config.data_path, train='weibo_train.csv',
                                            validation='weibo_val.csv',
                                            format='csv', skip_header=True,
                                            fields=[('label', LABEL), ('text', TEXT)])

    test = data.TabularDataset(path=weibo_config.test_csv, format='csv', skip_header=True,
                               fields=[('label', LABEL), ('text', TEXT)])
    # 构建词汇缓存目录
    cache = '.vector_cache'
    if not os.path.exists(cache):
        os.mkdir(cache)
    vectors = Vectors(name=weibo_config.pretrained_word_embedding, cache=cache)
    # 构建词汇表
    TEXT.build_vocab(train, max_size=weibo_config.vocab_size, vectors=vectors, unk_init=torch.Tensor.normal_)
    # TEXT.build_vocab(train, max_size=weibo_config.vocab_size) # 不使用预训练的词向量
    LABEL.build_vocab(train)

    # 构建迭代器
    train_iterator = data.Iterator(train, batch_size=64, device=weibo_config.device, sort_key=lambda x: len(x.text),
                                   sort_within_batch=True,
                                   repeat=False)
    val_iterator = data.Iterator(val, batch_size=64, device=weibo_config.device, sort_key=lambda x: len(x.text),
                                 sort_within_batch=True,
                                 repeat=False)
    test_iterator = data.Iterator(test, batch_size=64, device=weibo_config.device, sort_key=lambda x: len(x.text),
                                  sort_within_batch=True,
                                  repeat=False)

    return TEXT, LABEL, train_iterator, val_iterator, test_iterator


def cnews_data_process():
    config = CnewsConfig()
    # 加载数据
    # 数据清洗
    # 划分训练集
    # 保存csv
    # 定义field
    LABEL = data.LabelField()
    # TEXT = data.Field(sequential=True, tokenize=tokenizer, batch_first=True)
    TEXT = data.Field(sequential=True, tokenize=tokenizer, include_lengths=True)
    # 构建数据集
    train, val, test = data.TabularDataset.splits(path=config.csv_path, train='train.csv',
                                                  validation='val.csv', test='test.csv', skip_header=True, format='csv',
                                                  fields=[('label', LABEL), ('text', TEXT)])

    # 构建词汇缓存目录
    cache = '.vector_cache'
    if not os.path.exists(cache):
        os.mkdir(cache)
    vectors = Vectors(name=config.pretrained_word_embedding, cache=cache)
    # 构建词汇表
    TEXT.build_vocab(train, max_size=config.vocab_size, vectors=vectors, unk_init=torch.Tensor.normal_)
    # TEXT.build_vocab(train, max_size=weibo_config.vocab_size) # 不使用预训练的词向量
    LABEL.build_vocab(train)
    print(LABEL.vocab.stoi)
    # 构建迭代器

    train_iterator = data.Iterator(train, batch_size=64, device=config.device, sort_key=lambda x: len(x.text),
                                   sort_within_batch=True,
                                   repeat=False)
    val_iterator = data.Iterator(val, batch_size=64, device=config.device, sort_key=lambda x: len(x.text),
                                 sort_within_batch=True,
                                 repeat=False)
    test_iterator = data.Iterator(test, batch_size=128, device=config.device, sort_key=lambda x: len(x.text),
                                  sort_within_batch=True,
                                  repeat=False)

    return TEXT, LABEL, train_iterator, val_iterator, test_iterator


def bert_data_process():
    config = BertConfig()
    tokenizer = bert_tokenizer_init()  # 这里bert_model的路径写死在函数里了，如果用其他预训练的bert model，到函数里面进行修改
    print(f'bert字的个数：{len(tokenizer.vocab)}')
    # cls eos pad unk
    init_token = tokenizer.cls_token
    eos_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token
    unk_token = tokenizer.unk_token
    # cls eos pad unk 对应的下标
    init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
    eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)
    pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
    unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)

    TEXT = data.Field(batch_first=True, include_lengths=True,
                      use_vocab=False,
                      tokenize=tokenize_and_cut,
                      preprocessing=tokenizer.convert_tokens_to_ids,
                      init_token=init_token_idx,
                      eos_token=eos_token_idx,
                      pad_token=pad_token_idx,
                      unk_token=unk_token_idx)

    LABEL = data.LabelField()

    # 构建数据集
    train, val, test = data.TabularDataset.splits(path=config.csv_path, train='train.csv',
                                                  validation='val.csv', test='test.csv', skip_header=True, format='csv',
                                                  fields=[('label', LABEL), ('text', TEXT)])

    print(f"Number of training examples: {len(train)}")
    print(f"Number of validation examples: {len(val)}")
    print(f"Number of testing examples: {len(test)}")

    print(vars(train.examples[6]))
    tokens = tokenizer.convert_ids_to_tokens(vars(train.examples[6])['text'])
    LABEL.build_vocab(train)
    print(LABEL.vocab.stoi)

    print(tokens)
    # 构建迭代器

    train_iterator = data.Iterator(train, batch_size=64, device=config.device, sort_key=lambda x: len(x.text),
                                   sort_within_batch=True,
                                   repeat=False)
    val_iterator = data.Iterator(val, batch_size=64, device=config.device, sort_key=lambda x: len(x.text),
                                 sort_within_batch=True,
                                 repeat=False)
    test_iterator = data.Iterator(test, batch_size=128, device=config.device, sort_key=lambda x: len(x.text),
                                  sort_within_batch=True,
                                  repeat=False)

    return TEXT, LABEL, train_iterator, val_iterator, test_iterator


if __name__ == '__main__':
    # cnews_data_process()
    bert_data_process()
