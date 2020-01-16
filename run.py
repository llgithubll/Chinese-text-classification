from NNModels.trainers import test, trainer, parameter_prepared
import os
import random
import numpy as np
import torch
from config import GlobalConfig

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def run():
    global_config = GlobalConfig()
    # global_config.data_config.dataset_name = 'weibo_senti_100k'
    global_config.data_config.dataset_name = 'cnews'
    global_config.model_config.model_name = 'GRU-ATT'   # rnn 模型可选：'LSTM','LSTM-ATT','GRU','GRU-ATT','RNN',''

    # 根据数据集确定是否进行多分类
    if global_config.data_config.dataset_name in ['weibo_senti_100k']:
        global_config.is_multiclassification = False
    else:
        global_config.is_multiclassification = True

    global_config.is_bert_embedding = False  # 是否使用预训练的bert embedding
    parameter_prepared(global_config)
    # 训练
    trainer(global_config)
    # 测试
    test(global_config)


if __name__ == '__main__':
    run()
