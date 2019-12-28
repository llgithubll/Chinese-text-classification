from utils import train_rnn, evaluate_rnn, epoch_time
import time
from data_process import weibo_data_process
from utils import predict_sentiment,count_parameters
from config import WeiboConfig
from torch import optim, nn
from NNModels.layers import BiLSTM,FastText
import torch

def parameter_prepared():
    weibo_config = WeiboConfig()
    # 获取数据清洗后的结果
    weibo_config.TEXT, weibo_config.LABEL, weibo_config.train_iterator, weibo_config.val_iterator, weibo_config.test_iterator = weibo_data_process()
    # 词汇表的大小
    weibo_config.input_dim = len(weibo_config.TEXT.vocab)
    weibo_config.pad_idx = weibo_config.TEXT.vocab.stoi[weibo_config.TEXT.pad_token]
    weibo_config.unk_idx = weibo_config.TEXT.vocab.stoi[weibo_config.TEXT.unk_token]

    # 输出预训练embedding的shape
    # pretrained_embeddings = weibo_config.TEXT.vocab.vectors
    # print(f'word embedding shape:{pretrained_embeddings.shape}')


    # 定义模型
    weibo_config.model = FastText(weibo_config.input_dim, weibo_config.embedding_dim, weibo_config.output_dim, weibo_config.pad_idx)
    # 优化器，损失
    weibo_config.optimizer = optim.Adam(weibo_config.model.parameters(), lr=1e-3)
    weibo_config.criterion = nn.BCEWithLogitsLoss()

    # 把词向量copy到模型
    # weibo_config.model.embedding.weight.data.copy_(pretrained_embeddings)
    # 把unknown 和 pad 向量设置为零
    weibo_config.model.embedding.weight.data[weibo_config.unk_idx] = torch.zeros(weibo_config.embedding_dim)
    weibo_config.model.embedding.weight.data[weibo_config.pad_idx] = torch.zeros(weibo_config.embedding_dim)
    print('word embedding:')
    print(weibo_config.model.embedding.weight.data)

    # 加载到GPU
    weibo_config.model = weibo_config.model.to(weibo_config.device)
    weibo_config.criterion = weibo_config.criterion.to(weibo_config.device)

    print(f'The model has {count_parameters(weibo_config.model):,} trainable parameters')
    return weibo_config


def weibo_trainer(weibo_config):

    best_valid_loss = float('inf')

    for epoch in range(weibo_config.epoch):

        start_time = time.time()

        train_loss, train_acc = train_rnn(weibo_config.model, weibo_config.train_iterator, weibo_config.optimizer, weibo_config.criterion)
        valid_loss, valid_acc = evaluate_rnn(weibo_config.model, weibo_config.val_iterator, weibo_config.criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(weibo_config.model.state_dict(), weibo_config.best_model)

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')




def weibo_test(weibo_config):

    # 加载模型 进行测试
    weibo_config.model.load_state_dict(torch.load(weibo_config.best_model))

    test_loss, test_acc = evaluate_rnn(weibo_config.model, weibo_config.test_iterator, weibo_config.criterion)

    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')


