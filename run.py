from NNModels.trainers import weibo_test,weibo_trainer
from config import WeiboConfig
weibo_config = WeiboConfig()
weibo_config,test_iterator = weibo_trainer(weibo_config)
weibo_test(weibo_config,test_iterator)
