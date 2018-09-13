#-*- coding: utf-8 -*-

import os, sys, io
import json
import numpy as np
import xml.etree.cElementTree as ET
from utils import config

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def check_files(dir):
    """
    check the file is existed.
    """
    if not os.path.exists(dir):
        return False
    else:
        return True


class data_loader():
    def __init__(self, dir, config):
        """
        Initliaze
        """
        if not check_files(dir):
            raise FileNotFoundError('%s can not be founded !' % dir)
        # with open('./data/Sogo2008.dat', 'r', encoding)
        self.root = ET.parse(dir).getroot()
        self.config = config
        # print(self.root.tag)

    def padding(self, x):
        """
        padding to the same char size(max in list)
        """
        ml = max([len(i) for i in x])
        return [i + [0] * (ml - len(i)) for i in x]

    def get_vocab(self, jsfile=None):
        """
        get all chars and more char
        (
            mask: 0,
            unk: 1,
            start: 2,
            end: 3
        )
        """
        # 是否已经有词典
        # 有读取词典文件
        # 没有则生成词典文件
        if jsfile and check_files(jsfile):
            print("load the exist jsfile! ")
            chars, self.id2char, self.char2id = json.load(open(jsfile))
            self.id2char = {int(i) : j for i, j in self.id2char.items()}
        else:
            chars = {}
            # i = 0
            for doc in self.root.findall('doc'):
                # print(doc.find('contenttitle').text)
                try:
                    # print(doc.find('content').text)

                    for w in doc.find('content').text:
                        # 字典中有则count 1， 否则为0
                        # print(w)
                        chars[w] = chars.get(w, 0) + 1
                    for w in doc.find('contenttitle').text:
                        chars[w] = chars.get(w, 0) + 1
                except:
                    pass
                    # print("{} error!".format(i))
                # i += 1
                # if i == 100:
                  #   break

            print('suceesfully!')
            # 小于最小个数则不要
            chars = {i : j for i, j in chars.items() if j >= self.config.min_count}
            self.id2char = {i + 4 : j for i, j in enumerate(chars)}
            self.char2id = {j : i  for i, j in self.id2char.items()}
            json.dump([chars, self.id2char, self.char2id], open(jsfile, 'w'))

        return chars

    def str2id(self, s, start_end=False):
        """
        from text(str) to index
        """
        # 如果有start和end 且最大个数不大于maxlen
        # print("text: %s" % s)
        if start_end:
            # 如果不存在char2id的话，则为unk
            ids = [self.char2id.get(c, 1) for c in s[:self.config.maxlen - 2]]
            ids = [2] + ids + [3]
        else:
            ids = [self.char2id.get(c, 1) for c in s[:self.config.maxlen]]

        return ids

    def id2str(self, ids):
        """
        from index to char
        """
        return ''.join([self.id2char.get(i, '') for i in ids])

    def get_data(self):
        """
        get text and title
        """
        x, y = [], []
        while True:
            for doc in self.root.findall('doc'):
                text = doc.find('content').text
                title = doc.find('contenttitle').text
                if text and title:
                    x.append(self.str2id(text))
                    y.append(self.str2id(title, start_end=True))

                if len(x) == self.config.batch_size:
                    X = np.array(self.padding(x))
                    Y = np.array(self.padding(y))
                    yield [X, Y], None
                    x, y = [], []


if __name__ == '__main__':
    data = data_loader('./data/Sogo2008.dat', config)
    chars = data.get_vocab('./data/vocab.json')
    for item in data.get_data(config):
        continue
