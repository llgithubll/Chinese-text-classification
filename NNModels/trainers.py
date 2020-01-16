from utils import epoch_time, count_parameters, freeze_bert_paramers, show_paramers_require_grad, train_log_file, \
    test_log_file
from utils import categorical_evaluate, binary_evaluate
from utils import categorical_train, binary_train
import time
from torch import optim, nn
from NNModels.layers import RnnModel, FastText, RNN, TextCNN, BERTGRUSentiment, BertLSTM, RnnModelAttention, Bert
import torch
from data_process import data_process


def parameter_prepared(config):
    # 获取数据清洗后的结果
    config.data_config.TEXT, config.data_config.LABEL, config.data_config.train_iterator, config.data_config.val_iterator, config.data_config.test_iterator = data_process(
        config)

    if config.is_multiclassification:
        config.model_config.output_dim = len(config.data_config.LABEL.vocab)
    else:
        config.model_config.output_dim = 1  # 二分类

    if not config.is_bert_embedding:
        # 词汇表的大小
        config.model_config.input_dim = len(config.data_config.TEXT.vocab)
        config.data_config.pad_idx = config.data_config.TEXT.vocab.stoi[config.data_config.TEXT.pad_token]
        config.data_config.unk_idx = config.data_config.TEXT.vocab.stoi[config.data_config.TEXT.unk_token]
        # 输出预训练embedding的shape
        pretrained_embeddings = config.data_config.TEXT.vocab.vectors
        print(f'word embedding shape:{pretrained_embeddings.shape}')

        # 定义模型
        if config.model_config.model_name in ['LSTM', 'GRU', 'RNN']:
            config.model_config.model = RnnModel(config.model_config.model_name, config.model_config.input_dim,
                                                 config.model_config.embedding_dim,
                                                 config.model_config.hidden_dim, config.model_config.output_dim,
                                                 config.model_config.n_layer,
                                                 config.model_config.bidirection, config.model_config.dropout,
                                                 config.data_config.pad_idx, batch_first=config.data_config.batch_first)
        elif config.model_config.model_name in ['LSTM-ATT', 'GRU-ATT', 'RNN-ATT']:
            config.model_config.model = RnnModelAttention(config.model_config.model_name, config.model_config.input_dim,
                                                          config.model_config.embedding_dim,
                                                          config.model_config.hidden_dim,
                                                          config.model_config.output_dim,
                                                          config.model_config.n_layer,
                                                          config.model_config.bidirection, config.model_config.dropout,
                                                          config.data_config.pad_idx, device=config.device,
                                                          batch_first=config.data_config.batch_first
                                                          , attention_size=config.model_config.attention_size)

        # 优化器，损失
        config.trainer_config.optimizer = optim.Adam(config.model_config.model.parameters(), lr=1e-3)
        if config.is_multiclassification:
            config.trainer_config.criterion = nn.CrossEntropyLoss()
        else:
            config.trainer_config.criterion = nn.BCEWithLogitsLoss()  # 二分类

        # 把词向量copy到模型
        config.model_config.model.embedding.weight.data.copy_(pretrained_embeddings)
        # 把unknown 和 pad 向量设置为零
        config.model_config.model.embedding.weight.data[config.data_config.unk_idx] = torch.zeros(
            config.model_config.embedding_dim)
        config.model_config.model.embedding.weight.data[config.data_config.pad_idx] = torch.zeros(
            config.model_config.embedding_dim)
        print('word embedding:')
        print(config.model_config.model.embedding.weight.data)


    else:

        # 定义模型
        if config.model_config.model_name == 'BERT-LSTM':
            config.model_config.model = BertLSTM(config.data_config.bert, config.model_config.hidden_dim,
                                                 config.model_config.output_dim, config.model_config.n_layer,
                                                 config.model_config.bidirection,
                                                 config.model_config.dropout)
        elif config.model_config.model_name in ['BERT']:
            config.model_config.model = Bert(config.data_config.bert_model_path, config.model_config.hidden_dim,
                                             config.model_config.output_dim)

        # 将预训练的bert的参数固定住。
        print(count_parameters(config.model_config.model))

        # freeze_bert_paramers(config.model_config.model)  # 是否把bert模型固定住
        # print(count_parameters(config.model_config.model))
        # show_paramers_require_grad(config.model_config.model)

        # 优化器，损失
        config.trainer_config.optimizer = optim.Adam(config.model_config.model.parameters(), lr=1e-3)

        if config.is_multiclassification:
            config.trainer_config.criterion = nn.CrossEntropyLoss()
        else:
            config.trainer_config.criterion = nn.BCEWithLogitsLoss()  # 二分类

    # 加载到GPU
    # 开启并行运算
    # config.model = torch.nn.DataParallel(config.model)
    config.model_config.model = config.model_config.model.to(config.device)
    config.trainer_config.criterion = config.trainer_config.criterion.to(config.device)

    print(f'The model has {count_parameters(config.model_config.model):,} trainable parameters')


def trainer(config):
    best_valid_loss = float('inf')
    log_file = train_log_file(config.trainer_config.log_dir, config.data_config.dataset_name,
                              config.model_config.model_name,
                              config.trainer_config.epoch)
    for epoch in range(config.trainer_config.epoch):

        start_time = time.time()
        if config.is_multiclassification:
            train_loss, train_acc = categorical_train(config.model_config.model, config.data_config.train_iterator,
                                                      config.trainer_config.optimizer, config.trainer_config.criterion)
            valid_loss, valid_acc = categorical_evaluate(config.model_config.model, config.data_config.val_iterator,
                                                         config.trainer_config.criterion)
        else:
            train_loss, train_acc = binary_train(config.model_config.model, config.data_config.train_iterator,
                                                 config.trainer_config.optimizer, config.trainer_config.criterion)
            valid_loss, valid_acc = binary_evaluate(config.model_config.model, config.data_config.val_iterator,
                                                    config.trainer_config.criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(config.model_config.model.state_dict(),
                       config.model_config.best_model_path + '{}_bestmodel.pt'.format(config.model_config.model_name))

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
        log_file.write(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n')
        log_file.write(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%\n')
        log_file.write(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%\n')
    log_file.close()


def test(config):
    # log 文件
    log_file = test_log_file(config.trainer_config.log_dir, config.data_config.dataset_name,
                             config.model_config.model_name,
                             config.trainer_config.epoch)
    # 加载模型 进行测试
    config.model_config.model.load_state_dict(torch.load(config.model_config.best_model))
    if config.is_multiclassification:
        test_loss, test_acc = categorical_evaluate(config.model_config.model, config.data_config.test_iterator,
                                                   config.trainer_config.criterion)
    else:
        test_loss, test_acc = binary_evaluate(config.model_config.model, config.data_config.test_iterator,
                                              config.trainer_config.criterion)

    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
    log_file.write(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
    log_file.close()
