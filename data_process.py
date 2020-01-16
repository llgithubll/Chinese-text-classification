# -*- coding: utf-8 -*-
# @Time    : 2019/12/27 7:22 下午
# @Author  : lizhen
# @FileName: data_process.py
from pytorch_transformers import BertTokenizer

from config import DataConfig
import pandas as pd
from utils import clean_line, split_train_test, tokenizer, generate_bigrams
from torchtext import data
from torchtext.vocab import Vectors
import torch
import os
from utils import tokenize_and_cut


def data_process(config):
    # 针对 不同的数据集做不同的数据处理
    if config.data_config.dataset_name == 'weibo_senti_100k':
        config.data_config.csv_dir = config.data_config.weibo_csv_dir
        print(os.path.join(config.data_config.weibo_rawdata_dir, config.data_config.dataset_name + '.csv'))
        if not os.path.exists(config.data_config.weibo_csv_dir):  # 如果csv文件不存在

            os.mkdir(config.data_config.weibo_csv_dir)

            # 加载数据
            raw_data = pd.read_csv(
                os.path.join(config.data_config.weibo_rawdata_dir, config.data_config.dataset_name + '.csv'))
            # 数据清洗
            raw_data['review'] = raw_data['review'].apply(clean_line)
            # 划分训练集、验证集、测试集
            train_val, test_data = split_train_test(raw_data, 0.7)
            train_data, val_data = split_train_test(train_val, 0.8)
            # 保存成csv
            train_data.to_csv(os.path.join(config.data_config.weibo_csv_dir, 'train.csv'), index=0, encoding="utf8")
            val_data.to_csv(os.path.join(config.data_config.weibo_csv_dir, 'val.csv'), index=0, encoding="utf8")
            test_data.to_csv(os.path.join(config.data_config.weibo_csv_dir, 'test.csv'), index=0, encoding="utf8")
    elif config.data_config.dataset_name == 'cnews':
        config.data_config.csv_dir = config.data_config.cnews_csv_dir
        if not os.path.exists(config.data_config.cnews_csv_dir):
            os.mkdir(config.data_config.cnews_csv_dir)

            # 加载数据
            train_data = pd.read_table(os.path.join(config.data_config.cnews_rawdata_dir, 'cnews.train.txt'),
                                       header=None)
            val_data = pd.read_table(os.path.join(config.data_config.cnews_rawdata_dir, 'cnews.val.txt'), header=None)
            test_data = pd.read_table(os.path.join(config.data_config.cnews_rawdata_dir, 'cnews.test.txt'), header=None)
            # 数据清洗
            train_data[1] = train_data[1].apply(clean_line)
            val_data[1] = val_data[1].apply(clean_line)
            test_data[1] = test_data[1].apply(clean_line)
            # 划分训练集
            # 保存csv
            train_data.to_csv(os.path.join(config.data_config.cnews_csv_dir, 'train.csv'), index=False, encoding='utf8')
            val_data.to_csv(os.path.join(config.data_config.cnews_csv_dir, 'val.csv'), index=False, encoding='utf8')
            test_data.to_csv(os.path.join(config.data_config.cnews_csv_dir, 'test.csv'), index=False, encoding='utf8')

    # 是否使用bert预训练embedding
    if config.is_bert_embedding:
        bert_tokenizer = BertTokenizer.from_pretrained(config.data_config.bert_model_path)
        print(f'bert字的个数：{len(bert_tokenizer.vocab)}')
        # cls eos pad unk
        init_token = bert_tokenizer.cls_token
        eos_token = bert_tokenizer.sep_token
        pad_token = bert_tokenizer.pad_token
        unk_token = bert_tokenizer.unk_token
        # cls eos pad unk 对应的下标
        init_token_idx = bert_tokenizer.convert_tokens_to_ids(init_token)
        eos_token_idx = bert_tokenizer.convert_tokens_to_ids(eos_token)
        pad_token_idx = bert_tokenizer.convert_tokens_to_ids(pad_token)
        unk_token_idx = bert_tokenizer.convert_tokens_to_ids(unk_token)
        TEXT = data.Field(batch_first=True, include_lengths=True,
                          use_vocab=False,
                          tokenize=tokenize_and_cut,
                          preprocessing=bert_tokenizer.convert_tokens_to_ids,
                          init_token=init_token_idx,
                          eos_token=eos_token_idx,
                          pad_token=pad_token_idx,
                          unk_token=unk_token_idx)

        # torchtext的LABEL TEXT
        if config.is_multiclassification:
            LABEL = data.LabelField()
        else:
            LABEL = data.LabelField(sequential=False, use_vocab=False, dtype=torch.float)

        # 构建数据集 batch_first = True for TextCNN
        train, val, test = data.TabularDataset.splits(path=config.data_config.csv_dir, train='train.csv',
                                                      validation='val.csv', test='test.csv', skip_header=True,
                                                      format='csv',
                                                      fields=[('label', LABEL), ('text', TEXT)])

        print(f"Number of training examples: {len(train)}")
        print(f"Number of validation examples: {len(val)}")
        print(f"Number of testing examples: {len(test)}")

        LABEL.build_vocab(train)
        print(LABEL.vocab.stoi)

    else:

        # torchtext的LABEL TEXT
        if config.is_multiclassification:
            LABEL = data.LabelField()
        else:
            LABEL = data.LabelField(sequential=False, use_vocab=False, dtype=torch.float)

        if config.model_config.model_name in ['LSTM', 'LSTM-ATT', 'GRU', 'GRU-ATT', 'RNN', 'RNN-ATT']:
            TEXT = data.Field(sequential=True, tokenize=tokenizer, batch_first=config.data_config.batch_first,
                              include_lengths=config.data_config.include_lengths)  # include_lengths=True for LSTM
        elif config.model_config.model_name == 'TextCNN':
            TEXT = data.Field(sequential=True, tokenize=tokenizer, batch_first=True)
        elif config.model_config.model_name == 'FasterText':
            TEXT = data.Field(sequential=True, tokenize=tokenizer, preprocessing=generate_bigrams,
                              include_lengths=True)  # FastText
        else:
            TEXT = data.Field(sequential=True, tokenize=tokenizer)

        # 构建数据集
        train, val, test = data.TabularDataset.splits(path=config.data_config.csv_dir, train='train.csv',
                                                      validation='val.csv', test='test.csv', skip_header=True,
                                                      format='csv',
                                                      fields=[('label', LABEL), ('text', TEXT)])
        print("Number of training examples: {}".format(len(train)))
        print("Number of validation examples: {}".format(len(val)))
        print("Number of testing examples: {}".format(len(test)))

        # 是否使用预训练的词向量
        if config.data_config.is_use_pretrained_embedding:
            # 构建词汇缓存目录
            cache = '.vector_cache'
            if not os.path.exists(cache):
                os.mkdir(cache)

            vectors = Vectors(name=config.data_config.pretrained_word_embedding, cache=cache)
            TEXT.build_vocab(train, max_size=config.data_config.vocab_size, vectors=vectors,
                             unk_init=torch.Tensor.normal_)
            LABEL.build_vocab(train)
            print(LABEL.vocab.stoi)

        else:
            TEXT.build_vocab(train, max_size=config.data_config.vocab_size)  # 不使用预训练的词向量
            LABEL.build_vocab(train)
            print(LABEL.vocab.stoi)

    # 构建迭代器
    train_iterator = data.Iterator(train, batch_size=config.data_config.batch_size, device=config.device, sort_key=lambda x: len(x.text),
                                   sort_within_batch=True,
                                   repeat=False)
    val_iterator = data.Iterator(val, batch_size=config.data_config.batch_size, device=config.device, sort_key=lambda x: len(x.text),
                                 sort_within_batch=True,
                                 repeat=False)
    test_iterator = data.Iterator(test, batch_size=config.data_config.batch_size, device=config.device, sort_key=lambda x: len(x.text),
                                  sort_within_batch=True,
                                  repeat=False)

    return TEXT, LABEL, train_iterator, val_iterator, test_iterator
