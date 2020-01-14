# -*- coding: utf-8 -*-
# @Time    : 2019/12/27 6:22 下午
# @Author  : lizhen
# @FileName: utils.py
import torch
import numpy as np
import re
import pandas as pd
from collections import Counter
import jieba
import os
import tqdm
from pytorch_transformers import BertTokenizer, BertModel


def generate_bigrams(x):
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x


def count_parameters(model):
    """统计模型参数"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]])


def binary_train(model, iterator, optimizer, criterion):
    """二分类训练"""
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iter(iterator):
        optimizer.zero_grad()
        text, text_lengths = batch.text
        predictions = model(text, text_lengths).squeeze(1)

        loss = criterion(predictions, batch.label)

        acc = binary_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def binary_evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def categorical_train(model, iterator, optimizer, criterion):
    """多分类训练"""
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        text, text_lenths = batch.text

        predictions = model(text, text_lenths)  # for LSTM
        # predictions = model(text).squeeze(1)

        loss = criterion(predictions, batch.label)

        acc = categorical_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def categorical_evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, text_lenths = batch.text
            predictions = model(text, text_lenths) # for LSTM
            # predictions = model(text).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = categorical_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    """计算一次epoch时间"""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len)).tolist()

    x_shuffle = [x[i] for i in indices]
    y_shuffle = [y[j] for j in indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def predict_sentiment(config, sentence, min_len=5):
    """对输入的句子进行预测"""
    # 加载模型 进行测试
    config.model.load_state_dict(torch.load(config.best_model))
    config.model.eval()
    tokenized = tokenizer(sentence)
    if len(tokenized) < min_len:  # 在textcnn中 句子的最大 大小 不能小于 所有卷积核中宽度最大的卷积核宽度
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    indexed = [config.TEXT.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(config.device)
    tensor = tensor.unsqueeze(0)
    length_tensor = torch.LongTensor(length)
    # prediction = torch.sigmoid(weibo_config.model(tensor, length_tensor))
    prediction = torch.sigmoid(config.model(tensor))
    return prediction.item()


def predict_class(config, sentence, min_len=5):
    """多分类"""
    # 加载模型 进行测试
    config.model.load_state_dict(torch.load(config.best_model))
    config.model.eval()
    tokenized = tokenizer(sentence)
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    indexed = [config.TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(config.device)
    tensor = tensor.unsqueeze(1)
    preds = config.model(tensor)
    max_preds = preds.argmax(dim=1)
    return max_preds.item()


def freeze_bert_paramers(model):
    for name, param in model.named_parameters():
        if name.startswith('bert'):
            param.requires_grad = False


def show_paramers_require_grad(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)





def tokenize_and_cut(sentence, max_input_length=512):
    tokenizer = BertTokenizer.from_pretrained('./data/bert_model')
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length - 2]
    return tokens


def get_vector(embeddings, word):
    """获取word对应的词向量
        embedding:预训练或者训练好的词向量
    """
    assert word in embeddings.stoi, f'*{word}* is not in the vocab!'
    return embeddings.vectors[embeddings.stoi[word]]


def closest_words(embeddings, vector, n=10):
    """选择与向量vector最接近的是个向量，可以与get_vector结合使用，获取某个词最相似的十个词"""
    distances = [(word, torch.dist(vector, get_vector(embeddings, word)).item()) for word in embeddings.itos]

    return sorted(distances, key=lambda w: w[1])[:n]


def write_embeddings(path, embeddings, vocab):
    """保存训练好的embedding"""
    with open(path, 'w') as f:
        for i, embedding in enumerate(tqdm(embeddings)):
            word = vocab.itos[i]
            # skip words with unicode symbols
            if len(word) != len(word.encode()):
                continue
            vector = ' '.join([str(i) for i in embedding.tolist()])
            f.write(f'{word} {vector}\n')


# ~~~~~~~~~~~~~~~~~~~~~~~~分割线：下面全部都是数据预处理需要用到的函数~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def clean_line(s):
    """
    :param s: 清洗中文语料格式
    :return:
    """
    rule = re.compile(u'[^a-zA-Z0-9\u4e00-\u9fa5"#$%&\'()*+,-.:;<=>@\\^_`{|}]+')
    s = re.sub(rule, '', s)
    s = re.sub('[、]+', '，', s)
    s = re.sub('\'', '', s)
    s = re.sub('[#]+', '，', s)
    s = re.sub('[?]+', '？', s)
    s = re.sub('[;]+', '，', s)
    s = re.sub('[,]+', '，', s)
    s = re.sub('[!]+', '！', s)
    s = re.sub('[.]+', '.', s)
    s = re.sub('[，]+', '，', s)
    s = re.sub('[。]+', '。', s)
    s = re.sub('[~]+', '~', s)
    return s


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~分割线，下面是我自己写的，噢，上面也是我自己写的，下面的没有用torchtext包~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def tokenizer(text, use_stop_words=False):  # create a tokenizer function
    """
    定义分词操作
    """
    if use_stop_words:
        stop_words = stopwordslist('./data/stop_words.txt')
        text_cut = jieba.cut(text, cut_all=False)
        result = []
        for item in text_cut:
            if item not in stop_words:
                result.append(item)
        return result
    else:
        result = list(jieba.cut(text))
    return result


# 分词，去停用词
def stopwordslist(filepath):
    """加载停用词文件"""
    stopwords = [line.strip() for line in open(filepath, 'rb').readlines()]
    return stopwords


def split_train_test(data, frac):
    # DataFrame.sample(n=None, frac=None, replace=False, weights=None, random_state=None, axis=None)
    # n是选取的条数，frac是选取的比例，replace是可不可以重复选，weights是权重，random_state是随机种子，axis为0是选取行，为1是选取列。

    train_data = data.sample(frac=frac, random_state=1024, axis=0)
    test_data = data[~data.index.isin(train_data.index)]
    return train_data, test_data


def build_vocab(contents, vocab_size):
    """对文本构建词汇表,并返回句子中的词的个数最大值：max_sentence_len"""
    all_data = []
    max_sentence_len = 0
    for content in contents:
        all_data.extend(content.split())
        if len(content) > max_sentence_len:
            max_sentence_len = len(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 2)
    words, freq = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>', '<unknown>'] + list(words)
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id, max_sentence_len


def data_to_ids(data, word_to_id, max_lenth=500):
    """把句子中的词转换成id表示"""
    contents, labels = data['cuted_review'], data['label']

    sentence_ids, label_id = [], []
    for i in range(len(contents)):
        sentence_ids.append([word_to_id[x] if x in word_to_id else word_to_id['<unknown>'] for x in contents.iloc[i]])
        label_id.append(labels.iloc[i])
    return sentence_ids, label_id


def to_words(content, words):
    """将id表示的内容转换为文字"""
    return (words[x] for x in content)


def pad_sentence_maxlen(sentence, max_len):
    """# 把句子pad成最大长度"""
    for i in range(len(sentence)):
        if len(sentence[i]) < max_len:
            for j in range(max_len - len(sentence[i])):
                sentence[i].append(0)
    return sentence


def save_cnews_csv(cnews_path):
    test = pd.read_table(os.path.join(cnews_path, 'cnews.test.txt'), sep='\t', names=['label', 'text'], header=None)
    train = pd.read_table(os.path.join(cnews_path, 'cnews.train.txt'), sep='\t', names=['label', 'text'], header=None)
    val = pd.read_table(os.path.join(cnews_path, 'cnews.val.txt'), sep='\t', names=['label', 'text'], header=None)
    # 数据清洗
    test['clean_text'] = test['text'].apply(clean_line)
    test = test.drop(columns=['text'])
    train['clean_text'] = train['text'].apply(clean_line)
    train = train.drop(columns=['text'])
    val['clean_text'] = val['text'].apply(clean_line)
    val = val.drop(columns=['text'])

    test.to_csv(os.path.join(cnews_path, 'csv/test.csv'), index=0, encoding="utf8")
    train.to_csv(os.path.join(cnews_path, 'csv/train.csv'), index=0, encoding="utf8")
    val.to_csv(os.path.join(cnews_path, 'csv/val.csv'), index=0, encoding="utf8")
