import keras
import numpy as np
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import Callback

from model.seq2seq import Seq2Seq
from data.data_loader import data_loader
import utils.config as config


data = data_loader('./data/Sogo2008.dat', config)
chars = data.get_vocab('./data/vocab.json')


def gen_titles(s, topk=3):
    xid = np.array([data.str2id(s)] * topk)
    yid = np.array([[2]] * topk)
    scores = [0] * topk
    # print(xid, yid, scores)
    for i in range(50):
        proba = model.predict([xid, yid])[:, i, 3:]
        # print(proba.shape)
        log_proba = np.log(proba + 1e-6)
        arg_topk = log_proba.argsort(axis=1)[:, -topk:]
        _yid = []
        _socres = []
        if i == 0:
            for j in range(topk):
                # print(yid, arg_topk)
                _yid.append(list(yid[j]) + [arg_topk[0][j] + 3])
                _socres.append(scores[j] + log_proba[0][arg_topk[0][j]])
        else:
            for j in range(len(xid)):
                for k in range(topk):
                    _yid.append(list(yid[j]) + [arg_topk[j][k] + 3])
                    _socres.append(scores[j] + log_proba[j][arg_topk[j][k]])
            _arg_topk = np.argsort(_socres)[-topk:]
            _yid = [_yid[k] for k in _arg_topk]
            _socres = [_socres[k] for k in _arg_topk]
        yid = []
        scores = []
        for k in range(len(xid)):
            if _yid[k][-1] == 3:
                return data.id2str(_yid[k])
            else:
                yid.append(_yid[k])
                scores.append(_socres[k])
        yid = np.array(yid)
    return data.id2str(yid[np.argmax(scores)])

s1 = u'夏天来临，皮肤在强烈紫外线的照射下，晒伤不可避免，因此，晒后及时修复显得尤为重要，否则可能会造成长期伤害。专家表示，选择晒后护肤品要慎重，芦荟凝胶是最安全，有效的一种选择，晒伤严重者，还请及时就医。'
s2 = u'8月28日，网络爆料称，华住集团旗下连锁酒店用户数据疑似发生泄露。从卖家发布的内容看，数据包含华住旗下汉庭、禧玥、桔子、宜必思等10余个品牌酒店的住客信息。泄露的信息包括华住官网注册资料、酒店入住登记的身份信息及酒店开房记录，住客姓名、手机号、邮箱、身份证号、登录账号密码等。卖家对这个约5亿条数据打包出售。第三方安全平台威胁猎人对信息出售者提供的三万条数据进行验证，认为数据真实性非常高。当天下午，华住集团发声明称，已在内部迅速开展核查，并第一时间报警。当晚，上海警方消息称，接到华住集团报案，警方已经介入调查。'

class Evaluate(Callback):
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        print (gen_titles(s1))
        print (gen_titles(s2))
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights('./model/best_model.weights')


evaluator = Evaluate()
model = Seq2Seq(config, chars).run()
model.compile(optimizer=Adam(1e-3)) # lr
model.fit_generator(data.get_data(), steps_per_epoch=1000, epochs=config.epochs, callbacks=[evaluator])





