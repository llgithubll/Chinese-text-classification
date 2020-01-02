from NNModels.trainers import test, trainer, parameter_prepared,bert_parameter_prepared
from utils import predict_sentiment, predict_class
import os
import random
import numpy as np
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def run():
    config = bert_parameter_prepared()
    # 训练
    trainer(config)
    # 测试
    test(config)


if __name__ == '__main__':
    run()
