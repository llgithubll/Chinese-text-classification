import torch
from utils import batch_iter
from pytorch_transformers import BertModel


class DataConfig():
    """数据集及预处理的相关配置"""

    def __init__(self):
        self.stop_words = './data/stop_words.txt'  # 停用词路径

        self.dataset_name = ''
        # 微博情感数据集
        self.weibo_rawdata_dir = './data/weibo'  # 数据集路径
        self.weibo_csv_dir = './data/weibo/csv'
        # cnews 十分类数据集
        self.cnews_rawdata_dir = './data/cnews'
        self.cnews_csv_dir = './data/cnews/develop_csv'

        self.vocab_size = 25000  # 词汇表大小

        self.is_use_pretrained_embedding = True  # 是否使用预训练的embedding
        self.pretrained_word_embedding = './data/word_embeddings/sgns.sogou.word'  # 预训练的embedding
        self.batch_size = 4

        self.bert_model_path = './data/bert_model'
        self.batch_first = True
        self.include_lengths = True  # for rnn model
        self.bert = BertModel.from_pretrained(self.bert_model_path)


class ModelConfig:
    """模型的配置"""

    def __init__(self):
        self.model_name = ''
        self.embedding_dim = 300
        self.hidden_dim = 256
        self.n_layer = 2
        self.n_filter = 100
        self.filter_sizes = [3, 4, 5]
        self.dropout = 0.5
        self.epoch = 5
        self.best_model_path = './best_models/'
        self.bidirection = True
        self.attention_size = 32


class TrainerConfig:
    """训练的相关配置"""

    def __init__(self):
        self.epoch = 50
        self.log_dir = './train_log'


class GlobalConfig:
    """全局配置"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_multiclassification = False
        self.is_bert_embedding = True  # 是否使用bert训练模型

        # 包含 数据配置、模型配置、训练配置
        self.data_config = DataConfig()
        self.model_config = ModelConfig()
        self.trainer_config = TrainerConfig()

#
# class BertConfig():
#     def __init__(self):
#         self.bert_model = './data/bert_model/'
#         self.bert = BertModel.from_pretrained(self.bert_model)
#         self.stop_words = './data/stop_words.txt'
#         self.csv_path = './data/cnews/csv/'
#         self.pretrained_word_embedding = './data/word_embeddings/sgns.sogou.word'
#         self.vocab_size = 60000
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.embedding_dim = 300
#         self.hidden_dim = 256
#         self.output_dim = 10
#         self.n_layer = 2
#         self.n_filter = 100
#         self.filter_sizes = [3, 4, 5]
#         self.dropout = 0.25
#         self.epoch = 5
#         self.best_model = './best_models/bilstm_model.pt'
#         self.bidirection = True
