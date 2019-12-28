from NNModels.models import RNN
from config import WeiboConfig
from data_process import weibo_data_process
from utils import count_parameters, train, evaluate, epoch_time
from torch import optim, nn
import time
import torch


def weibo_trainer(weibo_config):
    # 获取数据清洗后的结果
    TEXT, LABEL, train_iterator, val_iterator, test_iterator = weibo_data_process()
    # 词汇表的大小
    weibo_config.input_dim = len(TEXT.vocab)
    # 模型定义，输出模型大小
    weibo_config.model = RNN(weibo_config.input_dim, weibo_config.embedding_dim, weibo_config.hidden_dim,
                             weibo_config.output_dim)
    print(f'The model has {count_parameters(weibo_config.model):,} trainable parameters')
    # 优化器，损失
    weibo_config.optimizer = optim.Adam(weibo_config.model.parameters(), lr=1e-3)
    weibo_config.criterion = nn.BCEWithLogitsLoss()

    model = weibo_config.model.to(weibo_config.device)
    weibo_config.criterion = weibo_config.criterion.to(weibo_config.device)

    best_valid_loss = float('inf')

    for epoch in range(weibo_config.epoch):

        start_time = time.time()

        train_loss, train_acc = train(model, train_iterator, weibo_config.optimizer, weibo_config.criterion)
        valid_loss, valid_acc = evaluate(model, val_iterator, weibo_config.criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), weibo_config.best_model)

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    return weibo_config, test_iterator


def weibo_test(weibo_config, test_iterator):
    # 加载模型 进行测试
    weibo_config.model.load_state_dict(torch.load(weibo_config.best_model))

    test_loss, test_acc = evaluate(weibo_config.model, test_iterator, weibo_config.criterion)

    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
