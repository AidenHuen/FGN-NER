# encoding: utf-8

from pypinyin import pinyin, lazy_pinyin
from keras.preprocessing.sequence import pad_sequences


def get_id_to_char_id(word2id, id2char, char2id, maxlen):
    # 得到词id 到特征id的转变, id2id的值是列表
    word_id = []
    char_id_l = []
    for k, v in id2char.items():
        char_id = []
        for a in v:
            try:
                char_id.append(char2id[a])
            except Exception, e:
                char_id.append(0)

        word_id.append(k)
        char_id_l.append(char_id)

    char_id_l = pad_sequences(char_id_l, maxlen=maxlen)
    id2id = dict(zip(word_id, char_id_l))
    return id2id


# 得到word的id对应的部分，col为对应feature文件中的列，maxlen为特征最长长度
def get_id2char(word2id, col, maxlen):
    id2char = {}
    char2id = {}
    with open('features.txt', 'r') as f1:
        for line in f1.readlines():
            word = line.split("::")[0].strip()
            if word not in word2id.keys():
                continue
            id = word2id[word]

            char_l = line.split("::")[col].strip().decode("utf-8")
            id2char[id] = char_l
    f1.close()

    # radical2id
    ch_l = []
    for charl in id2char.values():
        for ch in charl:
            if ch not in ch_l:
                ch_l.append(ch)

    i = 1
    for w in ch_l:
        char2id[w] = i
        i = i + 1
    id2id_char = get_id_to_char_id(word2id, id2char, char2id, maxlen)
    return id2id_char, char2id


def get_id2radical(word2id):
    id2id_radical, radical2id = get_id2char(word2id, col=7, maxlen=7)
    return id2id_radical, radical2id


def get_id2rad(word2id):
    id2id_rad, rad2id = get_id2char(word2id, col=2, maxlen=1)
    return id2id_rad, rad2id

# 声调转换dict
pingce = {
        'ā': 1, 'á': 2, 'ǎ': 3, 'à':4,
        'ē': 1, 'é': 2, 'ě': 3, 'è':4,
        'ī': 1, 'í': 2, 'ǐ': 3, 'ì':4,
        'ō': 1, 'ó': 2, 'ǒ': 3, 'ò':4,
        'ū': 1, 'ú': 2, 'ǔ': 3, 'ù':4,
 }


# 得到word的id对应的拼音，最长部首长度为8
def get_id2pinyin(word2id, maxlen=8):
    id2pinyin = {}
    with open('features.txt', 'r') as f1:
        for line in f1.readlines():
            word = line.split("::")[0].strip()
            if word not in word2id.keys():
                continue
            word_id = word2id[word]

            pin = pinyin(word.decode('utf-8'))
            # id2pinyin[id] = pin[0][0]   # plan a

            # plan b
            flag = 0  # 记录一二三四声，无为0
            for a in pin[0][0]:
                if a in pingce.keys():
                    flag = pingce[a.encode('utf-8')]

            piny = lazy_pinyin(word.decode('utf-8'))

            id2pinyin[word_id] = piny[0] + str(flag)
    f1.close()

    # pinyin2id
    pinyin2id = {}
    pin_l = []
    for pin1 in id2pinyin.values():
        for p1 in pin1:
            if p1 not in pin_l:
                pin_l.append(p1)

    for i, w in enumerate(pin_l):
        pinyin2id[w] = i + 1

    # 代表声调
    pinyin2id['0'] = 30
    pinyin2id['1'] = 31
    pinyin2id['2'] = 32
    pinyin2id['3'] = 33
    pinyin2id['4'] = 34
    id2id_pinyin = get_id_to_char_id(word2id, id2pinyin, pinyin2id, maxlen)
    return id2id_pinyin, pinyin2id




