# encoding:utf-8

import numpy
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
from keras_bert import Tokenizer
import config
import pickle
from skimage import io, transform
import codecs
from bert_serving.client import BertClient
import utiles

para = config.para

def get_char2id(train_x, id2id, maxlen):
    char_l = []
    lose = [0]*maxlen
    for sentence in train_x:
        sent_l = []
        for word_id in sentence:
            try:
                # print(id2id[word_id])
                sent_l.append(id2id[word_id])
            except Exception, e:
                sent_l.append(lose)
        # sent_l.insert(0,lose)
        # sent_l.append(lose)

        char_l.append(sent_l)

    return char_l


def ceate_feature_pk(para):

    with open(para["data_pk_path"]) as f:
        train_x, train_y, val_x, val_y, word2id, tags, image = pickle.load(f)
    f.close()
    # step1: 建立特征和id对应的词典及 字id到特征id的词典  只需要跑一次
    id2id_radical, radical2id = utiles.get_id2radical(word2id)
    # print(radical2id.__len__())
    id2id_rad, rad2id = utiles.get_id2rad(word2id)
    # print(rad2id.__len__())
    id2id_pinyin, pinyin2id = utiles.get_id2pinyin(word2id)

    # step 2: 求出train_x等对应的特征train

    radical_train = get_char2id(train_x, id2id_radical, para["radical_max"])
    pinyin_train = get_char2id(train_x, id2id_pinyin, para["pinyin_max"])
    rad_train = get_char2id(train_x, id2id_rad, para["rad_max"])

    radical_val = get_char2id(val_x, id2id_radical, para["radical_max"])
    pinyin_val = get_char2id(val_x, id2id_pinyin, para["pinyin_max"])
    rad_val = get_char2id(val_x, id2id_rad, para["rad_max"])



    # step3 radical_train list -> numpy.ndarry
    rad_train = numpy.array(rad_train).reshape(len(train_x), -1)
    rad_val = numpy.array(rad_val).reshape(len(val_x), -1)
    radical_train = numpy.array(radical_train).reshape(len(train_x), -1)
    radical_val = numpy.array(radical_val).reshape(len(val_x), -1)
    pinyin_train = numpy.array(pinyin_train).reshape(len(train_x), -1)
    pinyin_val = numpy.array(pinyin_val).reshape(len(val_x), -1)
    print(pinyin_val.shape)
    print(radical_val.shape)
    print(rad_val.shape)
    pickle.dump((radical_train, radical_val, pinyin_train, pinyin_val, rad_train, rad_val), open(para["fea_pk_path"], 'wb'))
    pickle.dump((radical2id, pinyin2id, rad2id, id2id_radical, id2id_pinyin, id2id_rad),open(para["dict_pk_path"], "wb"))

def load_img(path):
    image = io.imread(path,as_gray=True)
    image = image[10:90, 10:90]
    image = transform.resize(image, (para["img_h"], para["img_w"]))
    image = numpy.reshape(image,(para["img_h"], para["img_w"], -1))
    return image

def load_img_embed(word2id, simple2tradition = {}):
    print("Load image_voc...")
    img_embed = numpy.zeros((len(word2id.keys())+1,para["img_h"],para["img_w"], 1),dtype="float32")
    num = 0.0
    all = 0.0
    for word in word2id.keys():
        if simple2tradition.keys().__len__()!=0:
            item = simple2tradition[word]
        else:
            item = word
        try:
                img = load_img(config.fold_path+"image/"+item+".gif")
                img_embed[word2id[word]] = img
                num += 1
                all += 1
                # print(img.shape)
        except:
            img = numpy.random.rand(para["img_h"], para["img_w"], 1)
            all += 1
        # if word == '餐':
        #     img = load_img(config.fold_path+"image/"+item+".png")
        img_embed[word2id[word]] = img
    print(img_embed.shape)
    print("Load image_voc finish!!")
    print("汉字图像覆盖率："+str(num/all))
    return img_embed

def cross_validation(X,Y,fold):
    val_X = []
    val_Y = []
    train_X = []
    train_Y = []
    step = int(X.__len__() / fold)
    for i in range(fold):
        if i != fold - 1:
            val_X.append(X[step * i:step * (i + 1)])
            val_Y.append(Y[step * i:step * (i + 1)])
        else:
            val_X.append(X[step * i:])
            val_Y.append(Y[step * i:])
    for i in range(fold):
        X_list = []
        Y_list = []
        for j in range(val_X.__len__()):
            if j != i:
                X_list.append(val_X[j])
                Y_list.append(val_Y[j])
        train_X.append(numpy.concatenate(X_list, axis=0))
        train_Y.append(numpy.concatenate(Y_list, axis=0))
    return train_X, train_Y, val_X, val_Y

def train_test_dev_preprocess():
    train = _parse_data(codecs.open(para["train_path"], 'r'), sep=para["sep"])
    test = _parse_data(codecs.open(para["test_path"], 'r'), sep=para["sep"])
    dev = _parse_data(codecs.open(para["dev_path"], 'r'), sep=para["sep"])
    # train_len = train.__len__()
    print("Load dataset finish!!")
    dataset = train+test+dev
    tags = get_tag(dataset)
    print(tags)
    print(train.__len__(), test.__len__(), dev.__len__(), dataset.__len__())
    word_counts = Counter(row[0].lower() for sample in dataset for row in sample)
    vocab = [w for w, f in iter(word_counts.items()) if f >= 1]
    word2id = dict((w, i + 1) for i, w in enumerate(vocab))

    train_X, train_Y = process_data(train, word2id, tags)
    dev_X, dev_Y = process_data(dev, word2id, tags)
    test_X, test_Y = process_data(test, word2id, tags)
    print(train_X.shape,train_Y.shape)

    print("create X,Y,word2id finish!!")
    if para["traditional_chinese"]:
        s2t = get_simple2traditional()
        img_embed = load_img_embed(word2id,simple2tradition=s2t)
        "using tradiction chinese image...."
    else:
        img_embed = load_img_embed(word2id)
        "using simple chinese image...."
    print(train_X.shape,train_Y.shape,test_X.shape,test_Y.shape)
    pickle.dump((train_X, train_Y,  test_X, test_Y, dev_X,dev_Y, word2id, tags, img_embed), open(para["data_pk_path"], "wb"))

def train_test_set_preprocess():
    train = _parse_data(codecs.open(para["train_path"], 'r'), sep=para["sep"])
    test = _parse_data(codecs.open(para["test_path"], 'r'), sep=para["sep"])
    # train = dic+train
    print("Load trainset,dataset finish!!")
    dataset = train+test
    tags = get_tag(dataset)
    print(tags)
    print(train.__len__(),test.__len__(),dataset.__len__())
    word_counts = Counter(row[0].lower() for sample in dataset for row in sample)
    vocab = [w for w, f in iter(word_counts.items()) if f >= 1]
    word2id = dict((w, i + 1) for i, w in enumerate(vocab))
    # print(word2id)
    train_X, train_Y = process_data(train, word2id, tags)
    test_X, test_Y = process_data(test, word2id, tags)
    print(tags)

    print(train_X.shape,train_Y.shape)

    print("create X,Y,word2id finish!!")
    if para["traditional_chinese"]:
        s2t = get_simple2traditional()
        img_embed = load_img_embed(word2id,simple2tradition=s2t)
        "using tradiction chinese image...."
    else:
        img_embed = load_img_embed(word2id)
        "using simple chinese image...."
    print(train_X.shape,train_Y.shape,test_X.shape,test_Y.shape)
    pickle.dump((train_X, train_Y,  test_X, test_Y, word2id, tags, img_embed), open(para["data_pk_path"], "wb"))

class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R

def seq_padding(X, maxlen ,padding=0):
    L = len(X)
    ML = maxlen
    return numpy.array([
        numpy.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])

def bert_trainsfer_to_id(X,tokenizer):
    X1 = []
    X2 = []
    for i in range(X.__len__()):


        if X[i].__len__()<para["max_len"]:
            text = X[i]

        else:
            text= X[i][:para["max_len"]]

        text = "".join(text).decode("utf-8")
        x1, x2 = tokenizer.encode(first=text)

        # x1_sent = pad_sequences(x1,maxlen=para["char_len"])
        # x1_sent = seq_padding(x1, maxlen=para["max_len"])
        # x2_sent = seq_padding(x2, maxlen=para["max_len"])
        X1.append(x1)
        X2.append(x2)
    X1 = numpy.array(X1)
    X2 = numpy.array(X2)
    X1 = pad_sequences(X1, maxlen=para["max_len"]+2,padding='post', truncating='post')
    X2 = pad_sequences(X2, maxlen=para["max_len"]+2,padding='post', truncating='post')
    # for i in X1:
    #     print(i)
    print("X1 shape:",X1.shape)
    print("X2_shape:",X2.shape)
    return X1,X2

def bert_y_preprocess(data,tags):
    y = [[tags.index(w[1]) for w in s] for s in data]
    for sent_tags in y:
        sent_tags.insert(0,len(tags)-1)
        sent_tags.append(len(tags)-1)
    y = pad_sequences(y, para["max_len"]+2, value=-1,padding='post', truncating='post')
    y = numpy.reshape(y, newshape=(y.shape[0],y.shape[1],1))
    return y

def   bert_data_preprocess():
    train = _parse_data(codecs.open(para["train_path"], 'r'), sep=para["sep"])
    test = _parse_data(codecs.open(para["test_path"], 'r'), sep=para["sep"])
    x_train = [[items[0] for items in sent] for sent in train]
    x_test = [[items[0] for items in sent] for sent in test]

    token_dict = {}
    with codecs.open(para["bert_path"] + 'vocab.txt', 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    tokenizer = OurTokenizer(token_dict)
    x1_train, x2_train = bert_trainsfer_to_id(x_train, tokenizer)
    x1_test, x2_test = bert_trainsfer_to_id(x_test, tokenizer)
    #
    dataset = train+test
    tags = get_tag(dataset)
    tags.append("None")
    # print(tags)
    #
    y_train = bert_y_preprocess(train,tags)
    # print(y_train)
    y_test = bert_y_preprocess(test,tags)

    pickle.dump((x1_train, x2_train, y_train,  x1_test, x2_test, y_test, tags), open(para["data_pk_path"], "wb"))


def bert_feature_pk():
    train = _parse_data(codecs.open(para["train_path"], 'r'), sep=para["sep"])
    # print(train)
    test = _parse_data(codecs.open(para["test_path"], 'r'), sep=para["sep"])
    x_train = [[items[0].lower() for items in sent] for sent in train]
    x_test = [[items[0].lower()  for items in sent] for sent in test]
    for sent in x_train:
        sent.append("[END]")
        sent.insert(0,"[CLS]")
    for sent in x_test:
        sent.append("[END]")
        sent.insert(0,"[CLS]")
    dataset = x_train+x_test
    word_counts = Counter(row for sample in dataset for row in sample)
    vocab = [w for w, f in iter(word_counts.items()) if f >= 1]
    word2id = dict((w, i + 1) for i, w in enumerate(vocab))
    train_X = process_data(x_train, word2id)
    test_X = process_data(x_test, word2id)

    print(train_X.shape)
    id2id_radical, radical2id = utiles.get_id2radical(word2id)
    id2id_rad, rad2id = utiles.get_id2rad(word2id)
    id2id_pinyin, pinyin2id = utiles.get_id2pinyin(word2id)

    # step 2: 求出train_x等对应的特征train
    radical_train = get_char2id(train_X, id2id_radical, para["radical_max"])
    pinyin_train = get_char2id(train_X, id2id_pinyin, para["pinyin_max"])
    rad_train = get_char2id(train_X, id2id_rad, para["rad_max"])
    radical_test = get_char2id(test_X, id2id_radical, para["radical_max"])
    pinyin_test = get_char2id(test_X, id2id_pinyin, para["pinyin_max"])
    rad_test = get_char2id(test_X, id2id_rad, para["rad_max"])

    rad_train = numpy.array(rad_train).reshape(len(train_X), -1)
    rad_test = numpy.array(rad_test).reshape(len(test_X), -1)
    radical_train = numpy.array(radical_train).reshape(len(train_X), -1)
    radical_test = numpy.array(radical_test).reshape(len(test_X), -1)
    pinyin_train = numpy.array(pinyin_train).reshape(len(train_X), -1)
    pinyin_test = numpy.array(pinyin_test).reshape(len(test_X), -1)

    print(radical_train.shape)
    print(radical_test.shape)
    print("create X,Y,word2id finish!!")
    if para["traditional_chinese"]:
        s2t = get_simple2traditional()
        img_embed = load_img_embed(word2id,simple2tradition=s2t)
        "using tradiction chinese image...."
    else:
        img_embed = load_img_embed(word2id)
        "using simple chinese image...."

    pickle.dump((train_X,test_X, radical_train, radical_test, pinyin_train, pinyin_test, rad_train, rad_test, img_embed), open(para["fea_pk_path"], 'wb'))
    pickle.dump((word2id,radical2id, pinyin2id, rad2id, id2id_radical, id2id_pinyin, id2id_rad),open(para["dict_pk_path"], "wb"))


def load_bert_repre():
    train = _parse_data(codecs.open(para["train_path"], 'r'), sep=para["sep"])
    test = _parse_data(codecs.open(para["test_path"], 'r'), sep=para["sep"])
    train = [[items[0] for items in sent] for sent in train]
    test = [[items[0] for items in sent] for sent in test]

    train_x = numpy.zeros(shape=(train.__len__(),para["max_len"],768),dtype="float32")
    test_x = numpy.zeros(shape=(test.__len__(),para["max_len"],768),dtype="float32")

    bc = BertClient()

    step = int(train.__len__()/256)+1
    for i in range(step):
        if i != step-1:
            x = bc.encode(train[i*256:(i+1)*256], is_tokenized=True)
            x = x[:,1:para["max_len"]+1]
            train_x[i*256:((i+1)*256)] = x
            # print(train_x[i*256:(i+1)*256])
        else:
            x = bc.encode(train[i*256:],is_tokenized=True)
            x = x[:,1:para["max_len"]+1]
            train_x[i*256:] = x
            # print(train_x[i*256:])

    step = int(test.__len__() / 256) + 1
    # print(step)
    for i in range(step):
        if i != step - 1:
            x = bc.encode(test[i * 256:(i + 1) * 256], is_tokenized=True)
            x = x[:, 1:para["max_len"]+1]
            test_x[i * 256:((i + 1) * 256)] = x
            print(test_x[i * 256:(i + 1) * 256])
        else:
            x = bc.encode(test[i * 256:], is_tokenized=True)
            x = x[:, 1:para["max_len"]+1]
            test_x[i * 256:] = x
            # print(test_x[i * 256:])
    return train_x, test_x

def load_path_bert(path,sep="\t"):

    test = _parse_data(codecs.open(path, 'r'), sep=sep)
    test = [[items[0] for items in sent] for sent in test]
    test_x = numpy.zeros(shape=(test.__len__(), para["max_len"], 768),dtype="float32")
    bc = BertClient()

    step = int(test.__len__() / 256) + 1
    print(step)
    for i in range(step):
        if i != step - 1:
            x = bc.encode(test[i * 256:(i + 1) * 256], is_tokenized=True)
            x = x[:, 1:para["max_len"]+1]
            test_x[i * 256:((i + 1) * 256)] = x
            # print(test_x[i * 256:(i + 1) * 256])
        else:
            x = bc.encode(test[i * 256:], is_tokenized=True)
            x = x[:, 1:para["max_len"]+1]
            test_x[i * 256:] = x
            # print(test_x[i * 256:])
    # pickle.dump(test_x, open("./data/bert-pku-seg.pk", "wb"))
    return test_x

def get_tag(data):
    tag = []
    for words in data:
        for word_tag in words:
            if word_tag[1] not in tag:
                tag.append(word_tag[1])
    return tag


def _parse_data(file_input,sep="\t"):
    rows = file_input.readlines()
    rows[0] = rows[0].replace('\xef\xbb\xbf', '')
    items = [row.strip().split(sep) for row in rows]
    # print(items)
    max_len = 0
    sents = []
    sent = []
    n = 0
    for item in items:

        if item.__len__() != 1:
            sent.append(item)
        else:
            if sent.__len__() > para["max_len"]:
                n += 1
                split_sent = []
                for i, item in enumerate(sent):
                    if item[0] in ["。",",","，","!","！","?","？", "、", "；"] and split_sent.__len__()>10:
                        split_sent.append(item)
                        if split_sent.__len__() < para["max_len"]:
                            # for item in split_sent:
                            #     if item[1] != "O":
                            #         sents.append(split_sent[:])
                            #         break
                            # print(" ".join([item[0] for item in split_sent]))
                            sents.append(split_sent[:])
                        # else:
                        #     for item in split_sent:
                        #         print item[0],
                        #     print ""
                        split_sent = []
                    else:
                        split_sent.append(item)

                    # if i == sent.__len__()-1 and split_sent.__len__() < config.max_len:
                    #     for item in split_sent:
                    #         sents[sents.__len__()-1].append(item)
                    #     split_sent = []
                # continue
            else:
                if sent.__len__() > 1:
                    sents.append(sent[:])
            sent = []
    print  ("over_maxlen_sentence_num:", n)
    return sents


def _process_data(data, vocab, chunk_tags, maxlen=None, onehot=False):
    if maxlen is None:
        maxlen = max(len(s) for s in data)
    word2idx = dict((w, i+1) for i, w in enumerate(vocab))
    x = [[word2idx.get(w[0].lower(), 1) for w in s] for s in data]  # set to <unk> (index 1) if not in vocab
    y_chunk = [[chunk_tags.index(w[1]) for w in s] for s in data]
    x = pad_sequences(x, maxlen, padding='post', truncating='post')  # left padding
    y_chunk = pad_sequences(y_chunk, maxlen, value=-1, padding='post', truncating='post')
    if onehot:
        y_chunk = numpy.eye(len(chunk_tags), dtype='float32')[y_chunk]
        # print(y_chunk)
    else:
        y_chunk = numpy.expand_dims(y_chunk, 2)
    return x, y_chunk, word2idx

def process_data(data,word2idx):
    x = [[word2idx.get(w, 1) for w in s] for s in data]  # set to <unk> (index 1) if not in vocab

    # y_chunk = [[chunk_tags.index(w[1]) for w in s] for s in data]
    x = pad_sequences(x, para["max_len"]+2,padding='post', truncating='post')  # left padding
    # y_chunk = pad_sequences(y_chunk, para["max_len"], value=-1,padding='post', truncating='post')

    # if onehot:
    #     y_chunk = numpy.eye(len(chunk_tags), dtype='float32')[y_chunk]
    #     # print(y_chunk)
    # else:
    #     y_chunk = numpy.expand_dims(y_chunk, 2)
    return x


def get_lengths(X):
    lengths = []
    for i in range(len(X)):
        length = 0
        for dim in X[i]:
            if dim != 0:
                length += 1
            else:
                break
        # print(length)
        lengths.append(length)
    return lengths

def create_bool_matrex(repre_dim,x):
    bool_x = numpy.zeros(shape=(x.shape[0], x.shape[1],repre_dim))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] != 0:
                bool_x[i,j,:] = 1.
    return bool_x

def creat_bool_x(x):
    bool_x = numpy.zeros(shape=(x.shape[0],x.shape[1]),dtype="int32")
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] != 0:
                bool_x[i][j] = 1

    return bool_x

def creat_bool_embed(dim):
    weight = numpy.zeros(shape=(2,dim))
    weight[1,:] = 1.0
    return weight

def load_embed_weight(word2id):
    embed_weight = numpy.zeros(shape=(len(word2id.keys())+1, para["embed_dim"]))
    char2vec = {}
    with open(para["embed_path"], "r") as f:
        rows = f.readlines()
        for row in rows:
            item = row.strip().split(" ", 1)
            char = item[0]
            # print(item)
            vec_str = item[1].split(" ")
            vec = [float(i) for i in vec_str]
            char2vec[char] = vec
    for word in word2id.keys():
        # print(word)
        vec = char2vec[word]
        embed_weight[word2id[word]] = numpy.array(vec)
    # print(embed_weight)
    return embed_weight

def get_simple2traditional():
    simple2traditional = {}
    with open(config.traditional_dict_path,"r") as f:
        rows = f.readlines()
        for row in rows:
            item = row.strip().split("	")
            simple2traditional[item[0]] = item[1]
    return simple2traditional



if __name__ == "__main__":

    bert_data_preprocess()
    bert_feature_pk()
