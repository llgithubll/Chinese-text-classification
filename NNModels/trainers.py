from utils import epoch_time
from utils import categorical_evaluate as evaluate
from utils import categorical_train as train
import time
from data_process import weibo_data_process,cnews_data_process,bert_data_process
from utils import predict_sentiment,count_parameters,freeze_bert_paramers,show_paramers_require_grad
from config import WeiboConfig,CnewsConfig,BertConfig
from torch import optim, nn
from NNModels.layers import BiLSTM,FastText,RNN,TextCNN,BERTGRUSentiment
import torch

def parameter_prepared():
    # config = WeiboConfig()
    config = CnewsConfig()
    # 获取数据清洗后的结果
    config.TEXT, config.LABEL, config.train_iterator, config.val_iterator, config.test_iterator = bert_data_process()
    # 词汇表的大小
    config.input_dim = len(config.TEXT.vocab)
    config.output_dim = len(config.LABEL.vocab)
    config.pad_idx = config.TEXT.vocab.stoi[config.TEXT.pad_token]
    config.unk_idx = config.TEXT.vocab.stoi[config.TEXT.unk_token]

    # 输出预训练embedding的shape
    pretrained_embeddings = config.TEXT.vocab.vectors
    print(f'word embedding shape:{pretrained_embeddings.shape}')


    # 定义模型
    # config.model = BiLSTM(config.input_dim,config.embedding_dim,config.hidden_dim,config.output_dim,config.n_layer,
    #                       config.bidirection,config.dropout, config.pad_idx)
    config.model = BERTGRUSentiment(config.bert,config.hidden_dim,config.output_dim,config.n_layer,config.bidirection,config.dropout)

    # 将预训练的bert的参数固定住。
    count_parameters(config.model)
    freeze_bert_paramers(config.model)
    count_parameters(config.model)
    show_paramers_require_grad(config.model)

    # 优化器，损失
    config.optimizer = optim.Adam(config.model.parameters(), lr=1e-3)
    # config.criterion = nn.BCEWithLogitsLoss() # 二分类
    config.criterion = nn.CrossEntropyLoss()

    # 把词向量copy到模型
    config.model.embedding.weight.data.copy_(pretrained_embeddings)
    # 把unknown 和 pad 向量设置为零
    config.model.embedding.weight.data[config.unk_idx] = torch.zeros(config.embedding_dim)
    config.model.embedding.weight.data[config.pad_idx] = torch.zeros(config.embedding_dim)
    print('word embedding:')
    print(config.model.embedding.weight.data)

    # 加载到GPU
    # # 开启并行运算
    # config.model = torch.nn.DataParallel(config.model)
    config.model = config.model.to(config.device)
    config.criterion = config.criterion.to(config.device)

    print(f'The model has {count_parameters(config.model):,} trainable parameters')
    return config

def bert_parameter_prepared():
    # config = WeiboConfig()
    config = BertConfig()
    # 获取数据清洗后的结果
    config.TEXT, config.LABEL, config.train_iterator, config.val_iterator, config.test_iterator = bert_data_process()

    # 定义模型
    config.model = BERTGRUSentiment(config.bert,config.hidden_dim,config.output_dim,config.n_layer,config.bidirection,config.dropout)

    # 将预训练的bert的参数固定住。
    print(count_parameters(config.model))
    freeze_bert_paramers(config.model)
    print(count_parameters(config.model))
    show_paramers_require_grad(config.model)

    # 优化器，损失
    config.optimizer = optim.Adam(config.model.parameters(), lr=1e-3)
    # config.criterion = nn.BCEWithLogitsLoss() # 二分类
    config.criterion = nn.CrossEntropyLoss()

    # 加载到GPU
    # # 开启并行运算
    # config.model = torch.nn.DataParallel(config.model)
    config.model = config.model.to(config.device)
    config.criterion = config.criterion.to(config.device)

    print(f'The model has {count_parameters(config.model):,} trainable parameters')
    return config

def trainer(config):

    best_valid_loss = float('inf')

    for epoch in range(config.epoch):

        start_time = time.time()

        train_loss, train_acc = train(config.model, config.train_iterator, config.optimizer, config.criterion)
        valid_loss, valid_acc = evaluate(config.model, config.val_iterator, config.criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(config.model.state_dict(), config.best_model)

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')





def test(config):

    # 加载模型 进行测试
    config.model.load_state_dict(torch.load(config.best_model))

    test_loss, test_acc = evaluate(config.model, config.test_iterator, config.criterion)

    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')


