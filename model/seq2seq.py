import numpy as np
import os
from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam


class ScaleShift(Layer):
    """
    using prior knowledge.
    y = w * x + t (scale and layer)
    """
    def __init__(self, **kwargs):
        super(ScaleShift, self).__init__(**kwargs)

    def build(self, input_shape):
        kernel_shape = (1, ) * (len(input_shape) -1) + (input_shape[-1], )
        # w的学习变量
        self.log_scale = self.add_weight(name='log_scale', shape=kernel_shape, initializer='zeros')
        # t的学习变量
        self.shift = self.add_weight(name='shift', shape=kernel_shape, initializer='zeros')

    def call(self, inputs):
        x_out = K.exp(self.log_scale) * inputs + self.shift
        return x_out


class Interact(Layer):
    def __init__(self, **kwargs):
        super(Interact, self).__init__(**kwargs)

    def build(self, input_shape):
        in_dim = input_shape[0][-1]
        out_dim = input_shape[1][-1]
        self.kernel = self.add_weight(name='kernel', shape=(in_dim, out_dim), initializer='glorot_normal')

    def call(self, inputs):
        """
        v: -> x
        q: -> y
        """
        q, v, v_mask = inputs
        k = v
        mv = K.max(v - (1. - v_mask) * 1e10, axis=1, keepdims=True) # 一篇文章的vector(maxpooling)
        mv = mv + K.zeros_like(q[:, :, :1]) # mv和q的dim0和dim1相等
        # attention : 乘法注意力机制
        # h(i) * w(a) * s[j]
        qw = K.dot(q, self.kernel)
        a = K.batch_dot(qw, k, [2, 2]) / 10.
        # mask机制
        a -= (1. - K.permute_dimensions(v_mask, [0, 2, 1])) * 1e10
        a = K.softmax(a)
        o = K.batch_dot(a, v, [2, 1])
        return K.concatenate([o, q, mv], 2)

    def compute_output_shape(self, input_shape):
        """
        compute the output shape
        """
        return (None, input_shape[0][1], input_shape[0][2] + input_shape[1][2] * 2)



class Seq2Seq():
    def __init__(self, config, chars, **kwargs):
        """
        Initliaze
        """
        self.chars = chars
        self.config = config
        self.input_x = Input(shape=(None, ))
        self.input_y = Input(shape=(None, ))
        # self.mask = Lambda(lambda x : K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))
        # self.y_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(self.input_y)
        self.embedding = Embedding(len(self.chars) + 4, self.config.char_size)
        # self.bildrectional_x = Bidirectional(CuDNNLSTM(self.config.char_size / 2, return_sequences=True))
        # self.bildrectional_y = CuDNNLSTM(self.config.char_size, return_sequences=True)

    def to_one_hot(self, x):
        """
        each batch to one_hot (doc) with mask processing
        """
        x, x_mask = x
        x = K.cast(x, 'int32')
        x = K.one_hot(x, len(self.chars) + 4)
        # 词袋
        x = K.sum(x_mask * x, 1, keepdims=True)
        x = K.cast(K.greater(x, 0.5), 'float32')
        return x

    def run(self):
        """
        forward processing
        """
        x, y= self.input_x, self.input_y
        # 生成x_mask（如果不出现的单词则为0， 出现为1）
        x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x)
        y_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(y)
        x_one_hot = Lambda(self.to_one_hot)([x, x_mask])
        # 计算x的先验经验
        x_prior = ScaleShift()(x_one_hot)
        x = self.embedding(x)
        y = self.embedding(y)
        x = Bidirectional(CuDNNLSTM(int(self.config.char_size / 2), return_sequences=True))(x)
        x = Bidirectional(CuDNNLSTM(int(self.config.char_size / 2), return_sequences=True))(x)
        y = CuDNNLSTM(self.config.char_size, return_sequences=True)(y)
        y = CuDNNLSTM(self.config.char_size, return_sequences=True)(y)
        # x = self.bildrectional_x(x)
        # y = self.bildrectional_y(y)
        # y = self.bildrectional_y(y)
        xy = Interact()([y, x, x_mask])
        xy = Dense(512, activation='relu')(xy)
        xy = Dense(len(self.chars) + 4)(xy)
        xy = Lambda(lambda x: (x[0] + x[1]) / 2)([xy, x_prior])
        xy = Activation('softmax')(xy)


        cross_entropy = K.sparse_categorical_crossentropy(self.input_y[:, 1:], xy[:, :-1])
        # 去掉padding的部分
        loss = K.sum(cross_entropy * y_mask[:, 1:, 0]) / K.sum(y_mask[:, 1:, 0])

        model= Model([self.input_x, self.input_y], xy)
        model.add_loss(loss)
        return model

