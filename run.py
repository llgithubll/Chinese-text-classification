from NNModels.trainers import weibo_test, weibo_trainer, parameter_prepared
from utils import predict_sentiment

def weibo_run():
    weibo_config = parameter_prepared()
    # # 训练
    # weibo_trainer(weibo_config)
    # # 测试
    # weibo_test(weibo_config)
    print(predict_sentiment(weibo_config,'辣鸡苹果电脑啊啊啊啊，算了还是用苹果电脑吧，性价比还不错'))



if __name__ == '__main__':
    weibo_run()
