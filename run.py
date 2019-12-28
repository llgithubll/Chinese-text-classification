from NNModels.trainers import test, trainer, parameter_prepared
from utils import predict_sentiment, predict_class


def run():
    config = parameter_prepared()
    # 训练
    trainer(config)
    # 测试
    test(config)
    # print(predict_sentiment(config,'辣鸡苹果电脑啊啊啊啊，算了还是用苹果电脑吧，性价比还不错'))


if __name__ == '__main__':
    run()
