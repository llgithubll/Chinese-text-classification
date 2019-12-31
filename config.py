import torch
from utils import batch_iter


class WeiboConfig():
    def __init__(self):
        self.stop_words = './data/stop_words.txt'
        self.raw_data = './data/weibo_senti_100k.csv'
        self.vocab_size = 60000
        self.pretrained_word_embedding = './data/word_embeddings/sgns.weibo.bigram-char'
        self.train_csv = './data/weibo_train.csv'
        self.val_csv = './data/weibo_val.csv'
        self.test_csv = './data/weibo_test.csv'
        self.data_path = './data/'
        self.batch_size = 64
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_dim = 300
        self.n_filter = 100  # 卷积核数量
        self.filter_sizes = [3, 4, 5]  # 卷积核尺寸
        self.hidden_dim = 256
        self.output_dim = 1
        self.n_layer = 2
        self.bidirection = True

        self.dropout = 0.5
        self.epoch = 5
        self.iterator = batch_iter
        self.best_model = './best_models/bilstm_model.pt'

class CnewsConfig():
    def __init__(self):
        self.stop_words = './data/stop_words.txt'
        self.csv_path = './data/cnews/csv/'
        self.pretrained_word_embedding = './data/word_embeddings/sgns.sogou.word'
        self.vocab_size = 60000
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_dim = 300
        self.hidden_dim = 256
        self.n_layer = 2
        self.n_filter = 100
        self.filter_sizes = [3,4,5]
        self.dropout = 0.5
        self.epoch = 5
        self.best_model = './best_models/bilstm_model.pt'
        self.bidirection = True




